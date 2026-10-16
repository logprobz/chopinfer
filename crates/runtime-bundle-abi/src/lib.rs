//! Draft V1 C ABI for independently compiled runtime bundles.
//!
//! Rust types stay on one side of this boundary. The C layouts in this crate
//! match `include/chop_bundle_v1.h` and use only fixed-width fields.

#![deny(unsafe_code)]

pub mod frontend;
pub mod frontend_document;
mod host_services;
mod loader;
mod trampoline;
pub mod weight_plan;

pub use host_services::{
    AuthenticatedComponentMetadataV1, CollectiveRequestV1, CompiledImageSpanV1,
    CustomAddressSpaceV1, CustomCompletionV1, CustomOperandV1, DeviceBufferSpanV1,
    HostServiceResultV1, HostServicesBackendV1, KernelArgumentV1, LaunchClusterV1, LaunchConfigV1,
    LentLaunchV1, LentRangeV1, LentStreamV1, LogitsRowSinkV1, OwnedDraftTokensV1, OwnedLogitsRowV1,
    OwnedLogitsViewV1, PhysicalKvLayoutV1, PhysicalKvRequestV1, RankArenaUploadV1,
    ResultSinkLimitsV1, StateRequestV1, TensorMapElementV1, TensorMapSwizzleV1, TensorMapV1,
    UnsupportedHostServicesBackendV1,
};
pub use loader::{BundleLibraryV1, BundleLoaderError, LoadedBundleInstanceV1};
pub use trampoline::call_weight_transform_v1;
pub use trampoline::{
    BUNDLE_PREFIX_STATE_V1_MAX_PAGES, BUNDLE_PREFIX_STATE_V1_MAX_TOKENS, CustomOperationV1,
    HostServicesTableV1, HostServicesV1, PrefixStateRequestV1, catch_panic_api_pointer_v1,
    catch_panic_callback_status_v1, catch_panic_create_result_v1, catch_panic_prefix_state_info_v1,
    catch_panic_status_v1, checked_u64_to_usize_v1, dispatch_bundle_commit_accepted_v1,
    dispatch_bundle_create_v1, dispatch_bundle_execute_v1, dispatch_bundle_prefix_state_v1,
    dispatch_custom_operation_v1, dispatch_weight_plan_v1, dispatch_weight_transform_v1,
};

use std::ffi::c_void;
use std::fmt;
use std::mem::size_of;

pub const BUNDLE_ABI_VERSION_V1: u32 = 1;
pub const BUNDLE_PROBE_AUTHORITY_ROLE_V1: &[u8] = b"chop.device-probe.authority.v1";
pub const BUNDLE_PROBE_AUTHORITY_BYTES_V1: &[u8] = b"chop-device-probe-authority-v1\n";

/// Maximum role bytes an authenticated component service may expose.
pub const BUNDLE_AUTHENTICATED_COMPONENT_ROLE_BYTES_MAX_V1: u64 = 4096;
/// Maximum payload bytes an authenticated component service may expose.
pub const BUNDLE_AUTHENTICATED_COMPONENT_BYTES_MAX_V1: u64 = 1 << 20;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleStatusV1(pub u32);

#[allow(non_upper_case_globals)]
impl BundleStatusV1 {
    pub const Success: Self = Self(0);
    pub const InvalidArgument: Self = Self(1);
    pub const Unsupported: Self = Self(2);
    pub const NotFound: Self = Self(3);
    pub const ResourceExhausted: Self = Self(4);
    pub const StaleHandle: Self = Self(5);
    pub const InvalidState: Self = Self(6);
    pub const HostError: Self = Self(7);
    pub const BundleError: Self = Self(8);
    pub const AbiMismatch: Self = Self(9);
    pub const Panic: Self = Self(10);
    pub const InternalError: Self = Self(11);

    #[must_use]
    pub const fn is_success(self) -> bool {
        self.0 == Self::Success.0
    }

    /// The V1 name of this status, or `None` for a code V1 does not define.
    #[must_use]
    pub const fn name(self) -> Option<&'static str> {
        Some(match self.0 {
            0 => "success",
            1 => "invalid argument",
            2 => "unsupported",
            3 => "not found",
            4 => "resource exhausted",
            5 => "stale handle",
            6 => "invalid state",
            7 => "host error",
            8 => "bundle error",
            9 => "ABI mismatch",
            10 => "panic",
            11 => "internal error",
            _ => return None,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleByteViewV1 {
    pub data: *const u8,
    pub length: u64,
}

impl BundleByteViewV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            data: std::ptr::null(),
            length: 0,
        }
    }

    /// Describes borrowed bytes for one ABI call.
    ///
    /// The view records the address of `bytes` without borrowing it, so only a
    /// call that outlives neither the borrow nor its own frame may pass it on.
    /// Nothing outside the ABI trampolines reads through it.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            Self::empty()
        } else {
            Self {
                data: bytes.as_ptr(),
                length: u64::try_from(bytes.len()).expect("byte view length fits u64"),
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleMutByteViewV1 {
    pub data: *mut u8,
    pub length: u64,
}

impl BundleMutByteViewV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            data: std::ptr::null_mut(),
            length: 0,
        }
    }

    /// Describes exclusively borrowed bytes for one ABI call.
    ///
    /// The same call-scoped rule as [`BundleByteViewV1::from_bytes`]
    /// applies, and only the ABI trampolines write through it.
    #[must_use]
    pub fn from_bytes(bytes: &mut [u8]) -> Self {
        if bytes.is_empty() {
            Self::empty()
        } else {
            Self {
                data: bytes.as_mut_ptr(),
                length: u64::try_from(bytes.len()).expect("byte view length fits u64"),
            }
        }
    }
}

macro_rules! opaque_handle_v1 {
    ($name:ident) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
        pub struct $name {
            pub raw: u64,
        }

        impl $name {
            #[must_use]
            pub const fn from_raw(raw: u64) -> Self {
                Self { raw }
            }

            #[must_use]
            pub const fn slot(self) -> u32 {
                self.raw as u32
            }

            #[must_use]
            pub const fn generation(self) -> u32 {
                (self.raw >> 32) as u32
            }

            #[must_use]
            pub const fn from_parts(generation: u32, slot: u32) -> Self {
                Self {
                    raw: ((generation as u64) << 32) | slot as u64,
                }
            }

            #[must_use]
            pub const fn has_generation(self) -> bool {
                self.slot() != 0 && self.generation() != 0
            }

            #[must_use]
            pub const fn is_valid(self) -> bool {
                self.has_generation()
            }

            pub fn validate(self) -> Result<(), BundleAbiError> {
                if self.has_generation() {
                    Ok(())
                } else {
                    Err(BundleAbiError::InvalidHandle)
                }
            }
        }
    };
}

opaque_handle_v1!(BundleHandleV1);
opaque_handle_v1!(BundleContextHandleV1);
opaque_handle_v1!(BundleMemoryPoolHandleV1);
opaque_handle_v1!(BundleDeviceBufferHandleV1);
opaque_handle_v1!(BundleRankArenaHandleV1);
opaque_handle_v1!(BundleCompiledImageHandleV1);
opaque_handle_v1!(BundleAuthenticatedComponentHandleV1);
opaque_handle_v1!(BundlePeerAccessHandleV1);
opaque_handle_v1!(BundleStreamHandleV1);
opaque_handle_v1!(BundleGraphHandleV1);
opaque_handle_v1!(BundleEventHandleV1);
opaque_handle_v1!(BundleCudaModuleHandleV1);
opaque_handle_v1!(BundleCudaSymbolHandleV1);
opaque_handle_v1!(BundlePhysicalKvHandleV1);
opaque_handle_v1!(BundleKvRequestHandleV1);
opaque_handle_v1!(BundleKvTransactionHandleV1);
opaque_handle_v1!(BundleStateHandleV1);
opaque_handle_v1!(BundleCollectiveHandleV1);

pub(crate) const fn struct_size_v1<T>() -> u32 {
    size_of::<T>() as u32
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleDeviceBufferViewV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub buffer: BundleDeviceBufferHandleV1,
    pub offset: u64,
    pub length: u64,
}

impl BundleDeviceBufferViewV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            reserved: 0,
            buffer: BundleDeviceBufferHandleV1::from_raw(0),
            offset: 0,
            length: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleRawCompiledImageViewV1 {
    pub struct_size: u32,
    pub rank: u32,
    pub buffer: BundleDeviceBufferViewV1,
    pub alignment: u64,
}

impl BundleRawCompiledImageViewV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            rank: 0,
            buffer: BundleDeviceBufferViewV1::empty(),
            alignment: 0,
        }
    }
}

/// The fixed metadata of one package-authenticated component.
///
/// The SHA-256 field contains the exact 64 lowercase ASCII hexadecimal bytes
/// pinned by the verified package manifest. The role and payload are copied
/// separately while the generation-checked component handle remains live.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleAuthenticatedComponentMetadataV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub role_length: u64,
    pub byte_length: u64,
    pub sha256: [u8; 64],
}

impl BundleAuthenticatedComponentMetadataV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            reserved: 0,
            role_length: 0,
            byte_length: 0,
            sha256: [0; 64],
        }
    }
}

/// The blocks one thread-block cluster may hold.
///
/// Eight is the portable maximum: a cluster of that size launches on every
/// architecture that forms clusters at all, and a larger one is an opt-in this
/// version does not carry. A launch asking for more is refused rather than
/// reduced to one that would have run.
pub const BUNDLE_LAUNCH_V1_MAX_CLUSTER_BLOCKS: u32 = 8;

/// One launch shape, with the cluster its blocks are grouped into.
///
/// `cluster_x`, `cluster_y`, and `cluster_z` are blocks per cluster, each at
/// least one. All three at one asks for no cluster, which is the ordinary
/// launch. Any other shape is exact: the blocks multiply out to at most
/// [`BUNDLE_LAUNCH_V1_MAX_CLUSTER_BLOCKS`], and each grid extent divides
/// by the cluster extent beside it. A host that cannot launch the cluster asked
/// for refuses the launch rather than dropping the cluster and launching the
/// blocks, which would be a different program reported as this one.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleLaunchConfigV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
    pub shared_memory_bytes: u32,
    pub cluster_x: u32,
    pub cluster_y: u32,
    pub cluster_z: u32,
}

impl BundleLaunchConfigV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            flags: 0,
            grid_x: 0,
            grid_y: 0,
            grid_z: 0,
            block_x: 0,
            block_y: 0,
            block_z: 0,
            shared_memory_bytes: 0,
            cluster_x: 0,
            cluster_y: 0,
            cluster_z: 0,
        }
    }
}

/// The launch attribute a route asks for with this bit in `flags`.
///
/// A host that launches the symbol must apply CUDA's programmatic stream
/// serialization to it. The bit governs when a dependent launch may begin, not
/// what the kernel computes, and a host that cannot apply it owes a refusal
/// rather than a launch that quietly drops it.
pub const BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION: u32 = 1;

/// The launch attribute a route asks for with this bit in `flags`.
///
/// A host that launches the symbol must launch it cooperatively (CUDA's
/// cooperative launch attribute, with the launch's cluster if it has one):
/// every block of the grid is resident at once, or the launch is refused. A
/// grid whose blocks wait on each other needs that to finish, and a host that
/// cannot give it owes a refusal rather than a plain launch.
pub const BUNDLE_LAUNCH_FLAG_V1_COOPERATIVE: u32 = 2;

/// Every launch flag ABI V1 defines. Any other bit is refused.
pub const BUNDLE_LAUNCH_FLAGS_V1_KNOWN: u32 =
    BUNDLE_LAUNCH_FLAG_V1_PROGRAMMATIC_STREAM_SERIALIZATION | BUNDLE_LAUNCH_FLAG_V1_COOPERATIVE;

/// The closed set of tensor-map element types. FP8 uses CUDA's UINT8 encoding.
pub const BUNDLE_TENSOR_MAP_ELEMENT_V1_U8: u32 = 1;
pub const BUNDLE_TENSOR_MAP_ELEMENT_V1_I32: u32 = 2;
pub const BUNDLE_TENSOR_MAP_ELEMENT_V1_BF16: u32 = 3;
pub const BUNDLE_TENSOR_MAP_ELEMENT_V1_F16: u32 = 4;
pub const BUNDLE_TENSOR_MAP_ELEMENT_V1_F32: u32 = 5;
pub const BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E4M3: u32 = 6;
pub const BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E5M2: u32 = 7;
pub const BUNDLE_TENSOR_MAP_ELEMENT_V1_U16: u32 = 8;
pub const BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN8B: u32 = 9;
pub const BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN16B: u32 = 10;
pub const BUNDLE_TENSOR_MAP_ELEMENT_V1_16U6_ALIGN16B: u32 = 11;

/// The closed set of tensor-map swizzles this version encodes.
pub const BUNDLE_TENSOR_MAP_SWIZZLE_V1_NONE: u32 = 0;
pub const BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_128: u32 = 1;
pub const BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_32: u32 = 2;
pub const BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_64: u32 = 3;
pub const BUNDLE_TENSOR_MAP_INTERLEAVE_V1_NONE: u32 = 0;
pub const BUNDLE_TENSOR_MAP_INTERLEAVE_V1_BYTES_16: u32 = 1;
pub const BUNDLE_TENSOR_MAP_INTERLEAVE_V1_BYTES_32: u32 = 2;
pub const BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_NONE: u32 = 0;
pub const BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_64: u32 = 1;
pub const BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_128: u32 = 2;
pub const BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_256: u32 = 3;
pub const BUNDLE_TENSOR_MAP_OOB_FILL_V1_NONE: u32 = 0;
pub const BUNDLE_TENSOR_MAP_OOB_FILL_V1_NAN_REQUEST_ZERO_FMA: u32 = 1;

/// The byte width of an unpacked element, or the packed storage group unit.
#[must_use]
pub const fn tensor_map_element_bytes_v1(element_type: u32) -> Option<u64> {
    match element_type {
        BUNDLE_TENSOR_MAP_ELEMENT_V1_U8
        | BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E4M3
        | BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E5M2
        | BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN16B
        | BUNDLE_TENSOR_MAP_ELEMENT_V1_16U6_ALIGN16B => Some(1),
        BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN8B => Some(0),
        BUNDLE_TENSOR_MAP_ELEMENT_V1_I32 | BUNDLE_TENSOR_MAP_ELEMENT_V1_F32 => Some(4),
        BUNDLE_TENSOR_MAP_ELEMENT_V1_BF16
        | BUNDLE_TENSOR_MAP_ELEMENT_V1_F16
        | BUNDLE_TENSOR_MAP_ELEMENT_V1_U16 => Some(2),
        _ => None,
    }
}

/// The byte width one swizzle tiles the innermost box dimension over.
#[must_use]
pub const fn tensor_map_swizzle_bytes_v1(swizzle: u32) -> Option<u64> {
    match swizzle {
        BUNDLE_TENSOR_MAP_SWIZZLE_V1_NONE => Some(0),
        BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_128 => Some(128),
        BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_32 => Some(32),
        BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_64 => Some(64),
        _ => None,
    }
}

/// The largest box extent the tiled tensor-map encoding admits per dimension.
pub const BUNDLE_TENSOR_MAP_MAX_BOX_V1: u32 = 256;

/// The alignment a tiled tensor map requires of its global base address.
pub const BUNDLE_TENSOR_MAP_ADDRESS_ALIGNMENT_V1: u64 = 16;

/// One rank-1-to-5 tiled tensor map over a host-issued device-buffer span.
/// The C header states the geometry and packing rules.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleTensorMapV1 {
    pub struct_size: u32,
    pub rank: u32,
    pub element_type: u32,
    pub swizzle: u32,
    pub interleave: u32,
    pub l2_promotion: u32,
    pub oob_fill: u32,
    pub reserved: u32,
    pub buffer: BundleDeviceBufferViewV1,
    pub global_dims: [u64; 5],
    pub global_strides_bytes: [u64; 4],
    pub box_dims: [u32; 5],
    pub element_strides: [u32; 5],
}

impl BundleTensorMapV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            rank: 0,
            element_type: 0,
            swizzle: 0,
            interleave: 0,
            l2_promotion: 0,
            oob_fill: 0,
            reserved: 0,
            buffer: BundleDeviceBufferViewV1::empty(),
            global_dims: [0; 5],
            global_strides_bytes: [0; 4],
            box_dims: [0; 5],
            element_strides: [0; 5],
        }
    }
}

/// The slots one by-value device-pointer array carries.
///
/// Eight is the bound, not a shape: a fixed kernel ABI that takes a small
/// by-value table of device addresses declares how many of them it uses, and the
/// rest are null. Nothing here knows what a slot means. It equals
/// `kernel_core::contract::MAX_COMM_RANKS`, which bundle-runtime
/// asserts.
pub const BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS: u32 = 8;

/// A bounded by-value array of device addresses the host resolves.
///
/// Some fixed kernel ABIs take a small struct of device pointers by value. A
/// bundle cannot build one: it holds spans, not addresses. So it names the spans
/// in slot order and the host writes the addresses it has for them into the
/// argument, in that order. A slot naming the null handle at zero offset and
/// length is the null pointer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleDevicePointerArrayV1 {
    pub struct_size: u32,
    pub slot_count: u32,
    pub slots: [BundleDeviceBufferViewV1; BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS as usize],
}

impl BundleDevicePointerArrayV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            slot_count: 0,
            slots: [BundleDeviceBufferViewV1::empty();
                BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS as usize],
        }
    }
}

pub const BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_BUFFER: u32 = 1;
pub const BUNDLE_KERNEL_ARGUMENT_V1_U64: u32 = 2;
pub const BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES: u32 = 3;
pub const BUNDLE_KERNEL_ARGUMENT_V1_U32: u32 = 4;
pub const BUNDLE_KERNEL_ARGUMENT_V1_NULL_DEVICE_POINTER: u32 = 5;
pub const BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP: u32 = 6;
pub const BUNDLE_KERNEL_ARGUMENT_V1_I32: u32 = 7;
pub const BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY: u32 = 8;

/// The bytes one kernel argument may occupy.
///
/// An argument list is an array of these, passed by value, and a launch may
/// carry tens of them. The two large payloads are therefore named rather than
/// carried, which is what keeps this record small enough that the array is a
/// cheap thing to build and copy. The number is a ceiling on the V1 layout,
/// which the assertion below holds it under in both languages. It is not room
/// this record may grow into: see [`validate_kernel_argument_struct_size_v1`]
/// for why V1's kernel argument has no tail at all.
pub const BUNDLE_KERNEL_ARGUMENT_V1_MAX_BYTES: u32 = 96;

/// One kernel argument: a kind, and exactly the payload that kind selects.
///
/// This record is the one exception to the length rule every other V1 record
/// follows. It is an array element and no callback carries a stride, so its
/// `struct_size` must be V1's exactly rather than at least V1's, and V1 cannot
/// grow it. [`validate_kernel_argument_struct_size_v1`] is that rule.
///
/// `kind` is the discriminant and it is load bearing. Exactly one payload
/// belongs to each kind, and the fields no kind selects carry nothing:
///
/// | Kind | Payload |
/// | --- | --- |
/// | `DEVICE_BUFFER` | `buffer` |
/// | `U64`, `U32`, `I32` | the scalar of that width |
/// | `NULL_DEVICE_POINTER` | nothing |
/// | `OPAQUE_BYTES` | `opaque` |
/// | `TENSOR_MAP` | `tensor_map`, nonnull |
/// | `DEVICE_POINTER_ARRAY` | `pointer_array`, nonnull |
///
/// `tensor_map` and `pointer_array` point at records the caller keeps valid for
/// the length of the receiving call and no longer, exactly as a byte view is.
/// The record a kind selects must be nonnull, and each pointer-bearing field a
/// kind does not select must be absent. A record carrying a payload its kind
/// does not name says two things at once, so it is refused rather than read as
/// whichever one the reader happened to look at first. `buffer` is a handle and
/// two offsets rather than a pointer, so it is read only for `DEVICE_BUFFER` and
/// decides nothing elsewhere.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleKernelArgumentV1 {
    pub struct_size: u32,
    pub kind: u32,
    pub buffer: BundleDeviceBufferViewV1,
    pub scalar_u64: u64,
    pub scalar_u32: u32,
    pub scalar_i32: i32,
    pub opaque: BundleByteViewV1,
    pub tensor_map: *const BundleTensorMapV1,
    pub pointer_array: *const BundleDevicePointerArrayV1,
}

const _: () = {
    assert!(
        size_of::<BundleKernelArgumentV1>() <= BUNDLE_KERNEL_ARGUMENT_V1_MAX_BYTES as usize,
        "a V1 kernel argument must stay inside its declared size gate"
    );
};

impl BundleKernelArgumentV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            kind: 0,
            buffer: BundleDeviceBufferViewV1::empty(),
            scalar_u64: 0,
            scalar_u32: 0,
            scalar_i32: 0,
            opaque: BundleByteViewV1::empty(),
            tensor_map: std::ptr::null(),
            pointer_array: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundlePhysicalKvLayoutV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub rank_count: u64,
    pub page_bytes: u64,
    pub page_alignment: u64,
    pub page_tokens: u64,
    pub state_lifetime: u64,
    pub layout_id: BundleByteViewV1,
}

impl BundlePhysicalKvLayoutV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            flags: 0,
            rank_count: 0,
            page_bytes: 0,
            page_alignment: 0,
            page_tokens: 0,
            state_lifetime: 0,
            layout_id: BundleByteViewV1::empty(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundlePhysicalKvRequestV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub layout: BundlePhysicalKvLayoutV1,
    pub page_capacity: u64,
    pub request_capacity: u64,
}

impl BundlePhysicalKvRequestV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            flags: 0,
            layout: BundlePhysicalKvLayoutV1::empty(),
            page_capacity: 0,
            request_capacity: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleStateRequestV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub byte_length: u64,
    pub alignment: u64,
    pub state_id: BundleByteViewV1,
}

impl BundleStateRequestV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            flags: 0,
            byte_length: 0,
            alignment: 0,
            state_id: BundleByteViewV1::empty(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleCollectiveRequestV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub rank_count: u64,
    pub semantic_contract: BundleByteViewV1,
}

impl BundleCollectiveRequestV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            flags: 0,
            rank_count: 0,
            semantic_contract: BundleByteViewV1::empty(),
        }
    }
}

/// The bit a custom operand sets when its operation writes to that span.
pub const BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE: u32 = 1;

/// Every custom-operand flag ABI V1 defines. Any other bit is refused.
pub const BUNDLE_CUSTOM_OPERAND_FLAGS_V1_KNOWN: u32 = BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE;

/// One device-buffer span a bundle asks the host to lend a custom operation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleCustomOperandV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub buffer: BundleDeviceBufferViewV1,
}

impl BundleCustomOperandV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            flags: 0,
            buffer: BundleDeviceBufferViewV1::empty(),
        }
    }
}

/// The closed set of address spaces a lent custom launch can belong to.
pub const BUNDLE_CUSTOM_ADDRESS_SPACE_V1_DEVICE: u32 = 1;
pub const BUNDLE_CUSTOM_ADDRESS_SPACE_V1_HOST: u32 = 2;

/// The raw values one host lends a custom operation for exactly one call.
///
/// `operand_addresses` and `operand_lengths` each hold `operand_count` entries
/// in operand order, and both are null exactly when the count is zero.
/// `stream` is the host's own stream, already current with its context.
///
/// Nothing here may be retained. The record, both arrays, the stream, and every
/// address stop being the operation's the moment its callback returns. The host
/// waited on that stream before the callback, and waits again afterwards when
/// the operation said it may enqueue, so nothing it ordered outlives the call
/// or the buffers it was lent.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleCustomLaunchV1 {
    pub struct_size: u32,
    pub address_space: u32,
    pub operand_count: u64,
    pub stream: *mut c_void,
    pub operand_addresses: *const u64,
    pub operand_lengths: *const u64,
}

impl BundleCustomLaunchV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            address_space: 0,
            operand_count: 0,
            stream: std::ptr::null_mut(),
            operand_addresses: std::ptr::null(),
            operand_lengths: std::ptr::null(),
        }
    }
}

/// The bit a custom operation sets when it may leave work on the lent stream.
///
/// A call without it is synchronous: everything the operation started is
/// finished by the time its callback returns. A call with it is asynchronous,
/// and the host waits for what it enqueued before returning.
pub const BUNDLE_CUSTOM_OPERATION_FLAG_V1_MAY_ENQUEUE: u32 = 1;

/// Every custom-operation flag ABI V1 defines. Any other bit is refused.
pub const BUNDLE_CUSTOM_OPERATION_FLAGS_V1_KNOWN: u32 = BUNDLE_CUSTOM_OPERATION_FLAG_V1_MAY_ENQUEUE;

/// The one custom-operation code ABI V1 reserves.
///
/// An entry that contained a fault of its own reports it with this value rather
/// than letting the fault cross the boundary. A bundle numbers its own codes
/// with anything else.
pub const BUNDLE_CUSTOM_OPERATION_V1_CONTAINED_FAULT: i32 = i32::MIN;

pub type BundleCustomOperationFnV1 =
    unsafe extern "C" fn(*mut c_void, *const BundleCustomLaunchV1) -> i32;

pub type BundleRunCustomOperationFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleContextHandleV1,
    BundleStreamHandleV1,
    u32,
    BundleByteViewV1,
    *const BundleCustomOperandV1,
    u64,
    Option<BundleCustomOperationFnV1>,
    *mut c_void,
    *mut i32,
) -> BundleStatusV1;

pub type BundleLogFnV1 = unsafe extern "C" fn(*mut c_void, BundleByteViewV1) -> BundleStatusV1;
pub type BundleSelectContextFnV1 =
    unsafe extern "C" fn(*mut c_void, u32, i32, *mut BundleContextHandleV1) -> BundleStatusV1;
pub type BundleReleaseContextFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleContextHandleV1) -> BundleStatusV1;
pub type BundleSelectMemoryPoolFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleContextHandleV1,
    *mut BundleMemoryPoolHandleV1,
) -> BundleStatusV1;
pub type BundleSetMemoryPoolPeerAccessFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleMemoryPoolHandleV1,
    BundleContextHandleV1,
    u32,
) -> BundleStatusV1;
pub type BundleReleaseMemoryPoolFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleMemoryPoolHandleV1) -> BundleStatusV1;
pub type BundleAllocateDeviceBufferFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleMemoryPoolHandleV1,
    BundleStreamHandleV1,
    u64,
    u64,
    *mut BundleDeviceBufferHandleV1,
) -> BundleStatusV1;
pub type BundleReleaseDeviceBufferFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleDeviceBufferHandleV1) -> BundleStatusV1;

pub type BundleOpenCompiledImageFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleByteViewV1,
    *mut BundleCompiledImageHandleV1,
) -> BundleStatusV1;
pub type BundleUploadRankArenaFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleContextHandleV1,
    BundleMemoryPoolHandleV1,
    BundleStreamHandleV1,
    BundleCompiledImageHandleV1,
    u32,
    *mut BundleRankArenaHandleV1,
) -> BundleStatusV1;
/// One input for the optional batched rank upload service.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleRankArenaUploadV1 {
    pub context: BundleContextHandleV1,
    pub pool: BundleMemoryPoolHandleV1,
    pub stream: BundleStreamHandleV1,
    pub image: BundleCompiledImageHandleV1,
    pub rank: u32,
    pub reserved: u32,
}
pub type BundleUploadRankArenasFnV1 = unsafe extern "C" fn(
    *mut c_void,
    *const BundleRankArenaUploadV1,
    u64,
    *mut BundleRankArenaHandleV1,
) -> BundleStatusV1;
pub type BundleReleaseRankArenaFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleRankArenaHandleV1) -> BundleStatusV1;
pub type BundleResolveCompiledImageViewFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleCompiledImageHandleV1,
    BundleRankArenaHandleV1,
    u32,
    BundleByteViewV1,
    *mut BundleRawCompiledImageViewV1,
) -> BundleStatusV1;
pub type BundleReleaseCompiledImageFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleCompiledImageHandleV1) -> BundleStatusV1;
pub type BundleOpenAuthenticatedComponentFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleByteViewV1,
    *mut BundleAuthenticatedComponentHandleV1,
) -> BundleStatusV1;
pub type BundleInspectAuthenticatedComponentFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleAuthenticatedComponentHandleV1,
    *mut BundleAuthenticatedComponentMetadataV1,
) -> BundleStatusV1;
pub type BundleReadAuthenticatedComponentFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleAuthenticatedComponentHandleV1,
    BundleMutByteViewV1,
    BundleMutByteViewV1,
) -> BundleStatusV1;
pub type BundleReleaseAuthenticatedComponentFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleAuthenticatedComponentHandleV1) -> BundleStatusV1;

pub type BundleAcquirePeerAccessFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleContextHandleV1,
    BundleContextHandleV1,
    *mut BundlePeerAccessHandleV1,
) -> BundleStatusV1;
pub type BundleReleasePeerAccessFnV1 =
    unsafe extern "C" fn(*mut c_void, BundlePeerAccessHandleV1) -> BundleStatusV1;

pub type BundleCreateStreamFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleContextHandleV1,
    u32,
    *mut BundleStreamHandleV1,
) -> BundleStatusV1;
pub type BundleSelectStageStreamFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleContextHandleV1,
    u32,
    BundleByteViewV1,
    u32,
    *mut BundleStreamHandleV1,
) -> BundleStatusV1;
pub type BundleSynchronizeStreamFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleStreamHandleV1) -> BundleStatusV1;
pub type BundleReleaseStreamFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleStreamHandleV1) -> BundleStatusV1;

pub type BundleCreateEventFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleContextHandleV1,
    *mut BundleEventHandleV1,
) -> BundleStatusV1;
pub type BundleRecordEventFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleEventHandleV1, BundleStreamHandleV1) -> BundleStatusV1;
pub type BundleWaitEventFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleStreamHandleV1, BundleEventHandleV1) -> BundleStatusV1;
pub type BundleSynchronizeEventFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleEventHandleV1) -> BundleStatusV1;
pub type BundleReleaseEventFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleEventHandleV1) -> BundleStatusV1;

pub type BundleCopyHostToDeviceFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleStreamHandleV1,
    BundleDeviceBufferViewV1,
    BundleByteViewV1,
) -> BundleStatusV1;
pub type BundleCopyDeviceToHostFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleStreamHandleV1,
    BundleMutByteViewV1,
    BundleDeviceBufferViewV1,
) -> BundleStatusV1;
pub type BundleCopyDeviceToDeviceFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleStreamHandleV1,
    BundleDeviceBufferViewV1,
    BundleDeviceBufferViewV1,
) -> BundleStatusV1;
pub type BundleCopyPeerToPeerFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleStreamHandleV1,
    BundleContextHandleV1,
    BundleDeviceBufferViewV1,
    BundleContextHandleV1,
    BundleDeviceBufferViewV1,
) -> BundleStatusV1;
pub type BundleZeroDeviceBufferFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleStreamHandleV1,
    BundleDeviceBufferViewV1,
) -> BundleStatusV1;
pub type BundleWriteDevicePointerTableFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleStreamHandleV1,
    BundleDeviceBufferViewV1,
    *const BundleDevicePointerArrayV1,
) -> BundleStatusV1;

pub type BundleLoadCudaModuleFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleContextHandleV1,
    BundleByteViewV1,
    *mut BundleCudaModuleHandleV1,
) -> BundleStatusV1;
pub type BundleResolveCudaSymbolFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleCudaModuleHandleV1,
    BundleByteViewV1,
    *mut BundleCudaSymbolHandleV1,
) -> BundleStatusV1;
pub type BundleConfigureCudaSymbolFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleCudaSymbolHandleV1, u32) -> BundleStatusV1;
pub type BundleLaunchCudaSymbolFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleStreamHandleV1,
    BundleCudaSymbolHandleV1,
    *const BundleLaunchConfigV1,
    *const BundleKernelArgumentV1,
    u64,
) -> BundleStatusV1;
pub type BundleReleaseCudaSymbolFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleCudaSymbolHandleV1) -> BundleStatusV1;
pub type BundleReleaseCudaModuleFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleCudaModuleHandleV1) -> BundleStatusV1;

pub type BundleAllocatePhysicalKvFnV1 = unsafe extern "C" fn(
    *mut c_void,
    *const BundlePhysicalKvRequestV1,
    *mut BundlePhysicalKvHandleV1,
) -> BundleStatusV1;
pub type BundleCreateKvRequestFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundlePhysicalKvHandleV1,
    u64,
    *mut BundleKvRequestHandleV1,
) -> BundleStatusV1;
pub type BundlePrepareKvFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleKvRequestHandleV1,
    BundleByteViewV1,
    *mut BundleKvTransactionHandleV1,
) -> BundleStatusV1;
pub type BundleCommitKvFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleKvTransactionHandleV1) -> BundleStatusV1;
pub type BundleCancelKvFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleKvTransactionHandleV1) -> BundleStatusV1;
pub type BundleResolveKvViewFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleKvRequestHandleV1,
    u32,
    BundleByteViewV1,
    *mut BundleDeviceBufferViewV1,
) -> BundleStatusV1;
pub type BundleReleaseKvRequestFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleKvRequestHandleV1) -> BundleStatusV1;
pub type BundleReleasePhysicalKvFnV1 =
    unsafe extern "C" fn(*mut c_void, BundlePhysicalKvHandleV1) -> BundleStatusV1;
pub type BundleAcquireServingKvFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleByteViewV1, *mut u64) -> BundleStatusV1;
pub type BundleResolveServingKvPoolFnV1 =
    unsafe extern "C" fn(*mut c_void, u32, *mut BundleDeviceBufferViewV1) -> BundleStatusV1;

pub type BundleAllocateStateFnV1 = unsafe extern "C" fn(
    *mut c_void,
    *const BundleStateRequestV1,
    *mut BundleStateHandleV1,
) -> BundleStatusV1;
pub type BundleResolveStateViewFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleStateHandleV1,
    u32,
    *mut BundleDeviceBufferViewV1,
) -> BundleStatusV1;
pub type BundleReleaseStateFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleStateHandleV1) -> BundleStatusV1;

pub type BundleCreateCollectiveFnV1 = unsafe extern "C" fn(
    *mut c_void,
    *const BundleCollectiveRequestV1,
    *mut BundleCollectiveHandleV1,
) -> BundleStatusV1;
pub type BundleExecuteCollectiveFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleCollectiveHandleV1,
    BundleStreamHandleV1,
    BundleByteViewV1,
    *const BundleKernelArgumentV1,
    u64,
) -> BundleStatusV1;
pub type BundleReleaseCollectiveFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleCollectiveHandleV1) -> BundleStatusV1;

pub type BundleBeginGraphCaptureFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleStreamHandleV1) -> BundleStatusV1;
pub type BundleEndGraphCaptureFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleStreamHandleV1,
    *mut BundleGraphHandleV1,
) -> BundleStatusV1;
pub type BundleLaunchGraphFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleStreamHandleV1, BundleGraphHandleV1) -> BundleStatusV1;
pub type BundleReleaseGraphFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleGraphHandleV1) -> BundleStatusV1;

/// Enabled speculation bounds reported during bundle creation. Zero disables it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleRawSpeculationV1 {
    pub struct_size: u32,
    pub maximum_draft_tokens: u32,
    pub maximum_verify_tokens: u32,
}

impl BundleRawSpeculationV1 {
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            maximum_draft_tokens: 0,
            maximum_verify_tokens: 0,
        }
    }
}

pub type BundleQuerySpeculationFnV1 =
    unsafe extern "C" fn(*mut c_void, *mut BundleRawSpeculationV1) -> BundleStatusV1;

/// Batch limits the host schedules, reported during bundle creation so a
/// bundle can refuse limits it cannot execute. Zero means not reported.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleRawBatchLimitsV1 {
    pub struct_size: u32,
    /// Logits rows in one prefill phase that scores prompt rows.
    pub maximum_scored_rows: u32,
}

impl BundleRawBatchLimitsV1 {
    #[must_use]
    pub const fn unreported() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            maximum_scored_rows: 0,
        }
    }
}

pub type BundleQueryBatchLimitsFnV1 =
    unsafe extern "C" fn(*mut c_void, *mut BundleRawBatchLimitsV1) -> BundleStatusV1;

/// The process may be an MPS client: MPS clients share SMs spatially, so
/// another client's spinning grids can hold SMs this one's need.
pub const BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT: u32 = 1;
/// `max_active_clusters` is the driver's answer for the queried shape, so
/// zero there means no cluster of it can be resident; without this bit the
/// count is not reported.
pub const BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED: u32 = 2;
/// The device accepts cooperative launches
/// (`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`): a launch with
/// `BUNDLE_LAUNCH_FLAG_V1_COOPERATIVE` whose clusters fit
/// `max_active_clusters` runs every block at once.
pub const BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH: u32 = 4;

/// What the host's driver reports about keeping one symbol's clusters
/// resident on its context's device, for a bundle whose kernels spin on
/// flags other grids release and so must be resident all at once. Queried
/// during create; a zero field is not reported.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleRawClusterResidencyV1 {
    pub struct_size: u32,
    /// Streaming multiprocessors of the device.
    pub multiprocessor_count: u32,
    /// SMs the symbol's context may use: below `multiprocessor_count` under
    /// an MPS active-thread limit or an SM partition.
    pub usable_multiprocessors: u32,
    /// Clusters of the queried launch shape the device keeps resident at
    /// once (`cuOccupancyMaxActiveClusters`), with nothing else running;
    /// meaningful only with `BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED`.
    pub max_active_clusters: u32,
    /// `BUNDLE_CLUSTER_RESIDENCY_V1_*` bits.
    pub flags: u32,
}

impl BundleRawClusterResidencyV1 {
    #[must_use]
    pub const fn unreported() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            multiprocessor_count: 0,
            usable_multiprocessors: 0,
            max_active_clusters: 0,
            flags: 0,
        }
    }
}

pub type BundleQueryClusterResidencyFnV1 = unsafe extern "C" fn(
    *mut c_void,
    BundleCudaSymbolHandleV1,
    *const BundleLaunchConfigV1,
    *mut BundleRawClusterResidencyV1,
) -> BundleStatusV1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleHostServicesV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub log: Option<BundleLogFnV1>,
    pub select_context: Option<BundleSelectContextFnV1>,
    pub release_context: Option<BundleReleaseContextFnV1>,
    pub select_memory_pool: Option<BundleSelectMemoryPoolFnV1>,
    pub set_memory_pool_peer_access: Option<BundleSetMemoryPoolPeerAccessFnV1>,
    pub release_memory_pool: Option<BundleReleaseMemoryPoolFnV1>,
    pub allocate_device_buffer: Option<BundleAllocateDeviceBufferFnV1>,
    pub release_device_buffer: Option<BundleReleaseDeviceBufferFnV1>,
    pub open_compiled_image: Option<BundleOpenCompiledImageFnV1>,
    pub upload_rank_arena: Option<BundleUploadRankArenaFnV1>,
    pub release_rank_arena: Option<BundleReleaseRankArenaFnV1>,
    pub resolve_compiled_image_view: Option<BundleResolveCompiledImageViewFnV1>,
    pub release_compiled_image: Option<BundleReleaseCompiledImageFnV1>,
    pub open_authenticated_component: Option<BundleOpenAuthenticatedComponentFnV1>,
    pub inspect_authenticated_component: Option<BundleInspectAuthenticatedComponentFnV1>,
    pub read_authenticated_component: Option<BundleReadAuthenticatedComponentFnV1>,
    pub release_authenticated_component: Option<BundleReleaseAuthenticatedComponentFnV1>,
    pub acquire_peer_access: Option<BundleAcquirePeerAccessFnV1>,
    pub release_peer_access: Option<BundleReleasePeerAccessFnV1>,
    pub create_stream: Option<BundleCreateStreamFnV1>,
    pub select_stage_stream: Option<BundleSelectStageStreamFnV1>,
    pub synchronize_stream: Option<BundleSynchronizeStreamFnV1>,
    pub release_stream: Option<BundleReleaseStreamFnV1>,
    pub create_event: Option<BundleCreateEventFnV1>,
    pub record_event: Option<BundleRecordEventFnV1>,
    pub wait_event: Option<BundleWaitEventFnV1>,
    pub synchronize_event: Option<BundleSynchronizeEventFnV1>,
    pub release_event: Option<BundleReleaseEventFnV1>,
    pub copy_host_to_device: Option<BundleCopyHostToDeviceFnV1>,
    pub copy_device_to_host: Option<BundleCopyDeviceToHostFnV1>,
    pub copy_device_to_device: Option<BundleCopyDeviceToDeviceFnV1>,
    pub copy_peer_to_peer: Option<BundleCopyPeerToPeerFnV1>,
    pub zero_device_buffer: Option<BundleZeroDeviceBufferFnV1>,
    pub write_device_pointer_table: Option<BundleWriteDevicePointerTableFnV1>,
    pub load_cuda_module: Option<BundleLoadCudaModuleFnV1>,
    pub resolve_cuda_symbol: Option<BundleResolveCudaSymbolFnV1>,
    pub configure_cuda_symbol: Option<BundleConfigureCudaSymbolFnV1>,
    pub launch_cuda_symbol: Option<BundleLaunchCudaSymbolFnV1>,
    pub release_cuda_symbol: Option<BundleReleaseCudaSymbolFnV1>,
    pub release_cuda_module: Option<BundleReleaseCudaModuleFnV1>,
    pub allocate_physical_kv: Option<BundleAllocatePhysicalKvFnV1>,
    pub create_kv_request: Option<BundleCreateKvRequestFnV1>,
    pub prepare_kv: Option<BundlePrepareKvFnV1>,
    pub commit_kv: Option<BundleCommitKvFnV1>,
    pub cancel_kv: Option<BundleCancelKvFnV1>,
    pub resolve_kv_view: Option<BundleResolveKvViewFnV1>,
    pub release_kv_request: Option<BundleReleaseKvRequestFnV1>,
    pub release_physical_kv: Option<BundleReleasePhysicalKvFnV1>,
    pub allocate_state: Option<BundleAllocateStateFnV1>,
    pub resolve_state_view: Option<BundleResolveStateViewFnV1>,
    pub release_state: Option<BundleReleaseStateFnV1>,
    pub create_collective: Option<BundleCreateCollectiveFnV1>,
    pub execute_collective: Option<BundleExecuteCollectiveFnV1>,
    pub release_collective: Option<BundleReleaseCollectiveFnV1>,
    pub run_custom_operation: Option<BundleRunCustomOperationFnV1>,
    pub begin_graph_capture: Option<BundleBeginGraphCaptureFnV1>,
    pub end_graph_capture: Option<BundleEndGraphCaptureFnV1>,
    pub launch_graph: Option<BundleLaunchGraphFnV1>,
    pub release_graph: Option<BundleReleaseGraphFnV1>,
    pub acquire_serving_kv: Option<BundleAcquireServingKvFnV1>,
    pub resolve_serving_kv_pool: Option<BundleResolveServingKvPoolFnV1>,
    pub upload_rank_arenas: Option<BundleUploadRankArenasFnV1>,
    pub query_speculation_v1: Option<BundleQuerySpeculationFnV1>,
    pub query_batch_limits_v1: Option<BundleQueryBatchLimitsFnV1>,
    pub query_cluster_residency_v1: Option<BundleQueryClusterResidencyFnV1>,
}

impl BundleHostServicesV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            abi_version: BUNDLE_ABI_VERSION_V1,
            user_data: std::ptr::null_mut(),
            log: None,
            select_context: None,
            release_context: None,
            select_memory_pool: None,
            set_memory_pool_peer_access: None,
            release_memory_pool: None,
            allocate_device_buffer: None,
            release_device_buffer: None,
            open_compiled_image: None,
            upload_rank_arena: None,
            release_rank_arena: None,
            resolve_compiled_image_view: None,
            release_compiled_image: None,
            open_authenticated_component: None,
            inspect_authenticated_component: None,
            read_authenticated_component: None,
            release_authenticated_component: None,
            acquire_peer_access: None,
            release_peer_access: None,
            create_stream: None,
            select_stage_stream: None,
            synchronize_stream: None,
            release_stream: None,
            create_event: None,
            record_event: None,
            wait_event: None,
            synchronize_event: None,
            release_event: None,
            copy_host_to_device: None,
            copy_device_to_host: None,
            copy_device_to_device: None,
            copy_peer_to_peer: None,
            zero_device_buffer: None,
            write_device_pointer_table: None,
            load_cuda_module: None,
            resolve_cuda_symbol: None,
            configure_cuda_symbol: None,
            launch_cuda_symbol: None,
            release_cuda_symbol: None,
            release_cuda_module: None,
            allocate_physical_kv: None,
            create_kv_request: None,
            prepare_kv: None,
            commit_kv: None,
            cancel_kv: None,
            resolve_kv_view: None,
            release_kv_request: None,
            release_physical_kv: None,
            allocate_state: None,
            resolve_state_view: None,
            release_state: None,
            create_collective: None,
            execute_collective: None,
            release_collective: None,
            run_custom_operation: None,
            begin_graph_capture: None,
            end_graph_capture: None,
            launch_graph: None,
            release_graph: None,
            acquire_serving_kv: None,
            resolve_serving_kv_pool: None,
            upload_rank_arenas: None,
            query_speculation_v1: None,
            query_batch_limits_v1: None,
            query_cluster_residency_v1: None,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleCreateResultV1 {
    pub struct_size: u32,
    pub status: BundleStatusV1,
    pub handle: BundleHandleV1,
}

impl BundleCreateResultV1 {
    #[must_use]
    pub const fn success(handle: BundleHandleV1) -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            status: BundleStatusV1::Success,
            handle,
        }
    }

    #[must_use]
    pub const fn failure(status: BundleStatusV1) -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            status,
            handle: BundleHandleV1::from_raw(0),
        }
    }
}

/// One model-neutral logits row delivered before a transaction is finalized.
///
/// `logit_bits` contains little-endian IEEE 754 binary32 bit patterns. Its
/// length is exactly four times `logit_count`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleLogitsRowV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub request_id: u64,
    pub logit_count: u64,
    pub logit_bits: BundleByteViewV1,
}

impl BundleLogitsRowV1 {
    pub(crate) fn from_logit_bits(
        request_id: u64,
        logit_bits: &[u8],
    ) -> Result<Self, BundleAbiError> {
        if logit_bits.is_empty() || logit_bits.len() % 4 != 0 {
            return Err(BundleAbiError::InvalidLogitsRow);
        }
        let logit_count =
            u64::try_from(logit_bits.len() / 4).map_err(|_| BundleAbiError::InvalidLogitsRow)?;
        Ok(Self {
            struct_size: struct_size_v1::<Self>(),
            reserved: 0,
            request_id,
            logit_count,
            logit_bits: BundleByteViewV1::from_bytes(logit_bits),
        })
    }
}

pub type BundleEmitLogitsRowFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleLogitsRowV1) -> BundleStatusV1;

pub const BUNDLE_LOGITS_ELEMENT_V1_F32: u32 = 1;
pub const BUNDLE_LOGITS_ELEMENT_V1_BF16: u32 = 2;

/// Ordered rows in a host-owned device buffer. The live width excludes padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleLogitsViewV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub context: BundleContextHandleV1,
    pub rank: u32,
    pub element_type: u32,
    pub buffer: BundleDeviceBufferViewV1,
    pub row_stride_elements: u64,
    pub row_count: u32,
    pub live_width: u32,
    pub first_item_index: u32,
    pub reserved_tail: u32,
}

impl BundleLogitsViewV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            reserved: 0,
            context: BundleContextHandleV1::from_raw(0),
            rank: 0,
            element_type: 0,
            buffer: BundleDeviceBufferViewV1::empty(),
            row_stride_elements: 0,
            row_count: 0,
            live_width: 0,
            first_item_index: 0,
            reserved_tail: 0,
        }
    }
}

pub type BundleEmitLogitsViewFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleLogitsViewV1) -> BundleStatusV1;

/// Drafted tokens, encoded as little-endian u32 values in a borrowed byte view.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleDraftTokensV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub request_id: u64,
    pub token_count: u32,
    pub token_bytes: BundleByteViewV1,
}

pub type BundleEmitDraftTokensFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleDraftTokensV1) -> BundleStatusV1;

/// A bounded host-owned result channel for one `execute` call.
///
/// The bundle invokes `emit_logits_row` synchronously and retains neither this
/// record nor any row pointer after `execute` returns. The host copies row
/// bytes before its callback returns.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleExecuteResultSinkV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub user_data: *mut c_void,
    pub maximum_rows: u64,
    pub maximum_logits_per_row: u64,
    pub maximum_total_logits: u64,
    pub emit_logits_row: Option<BundleEmitLogitsRowFnV1>,
    pub emit_logits_view: Option<BundleEmitLogitsViewFnV1>,
    pub emit_draft_tokens: Option<BundleEmitDraftTokensFnV1>,
}

impl BundleExecuteResultSinkV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            reserved: 0,
            user_data: std::ptr::null_mut(),
            maximum_rows: 0,
            maximum_logits_per_row: 0,
            maximum_total_logits: 0,
            emit_logits_row: None,
            emit_logits_view: None,
            emit_draft_tokens: None,
        }
    }
}

/// The bundle-side result channel for one execute call.
///
/// A generated execute trampoline builds it from the native sink and lends it
/// to the bundle for that call only. The bundle emits rows from its own byte
/// slices, so it never builds or retains a pointer-backed row.
pub struct ExecuteResultSinkV1 {
    raw: BundleExecuteResultSinkV1,
    emitted_rows: u64,
    emitted_logits: u64,
}

impl ExecuteResultSinkV1 {
    pub(crate) fn new(raw: BundleExecuteResultSinkV1) -> Result<Self, BundleAbiError> {
        validate_execute_result_sink_v1(raw)?;
        Ok(Self {
            raw,
            emitted_rows: 0,
            emitted_logits: 0,
        })
    }

    #[must_use]
    pub const fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(
            self.raw.maximum_rows,
            self.raw.maximum_logits_per_row,
            self.raw.maximum_total_logits,
        )
    }

    #[must_use]
    pub const fn emitted_rows(&self) -> u64 {
        self.emitted_rows
    }

    #[must_use]
    pub const fn emitted_logits(&self) -> u64 {
        self.emitted_logits
    }

    /// Delivers one row of little-endian binary32 logit bits to the host.
    pub fn emit(&mut self, request_id: u64, logit_bits: &[u8]) -> Result<(), BundleAbiError> {
        let row = BundleLogitsRowV1::from_logit_bits(request_id, logit_bits)?;
        validate_execute_logits_row_v1(row)?;
        charge_result_capacity_v1(
            self.limits(),
            &mut self.emitted_rows,
            &mut self.emitted_logits,
            row.logit_count,
        )?;
        trampoline::invoke_emit_logits_row_v1(&self.raw, row)
    }

    /// Delivers a validated device view while the host owns its buffer.
    pub fn emit_device_view(&mut self, view: BundleLogitsViewV1) -> Result<(), BundleAbiError> {
        validate_execute_logits_view_v1(view)?;
        if (self.raw.struct_size as usize)
            < std::mem::offset_of!(BundleExecuteResultSinkV1, emit_logits_view)
                + std::mem::size_of_val(&self.raw.emit_logits_view)
        {
            return Err(BundleAbiError::MissingExecuteResultCallback);
        }
        charge_result_capacity_rows_v1(
            self.limits(),
            &mut self.emitted_rows,
            &mut self.emitted_logits,
            u64::from(view.row_count),
            u64::from(view.live_width),
        )?;
        trampoline::invoke_emit_logits_view_v1(&self.raw, view)
    }

    /// Delivers drafts before the logits rows that use them.
    pub fn emit_draft_tokens(
        &mut self,
        request_id: u64,
        tokens: &[u32],
    ) -> Result<(), BundleAbiError> {
        let token_count =
            u32::try_from(tokens.len()).map_err(|_| BundleAbiError::InvalidDraftTokens)?;
        if token_count == 0 || token_count > 1 << 20 {
            return Err(BundleAbiError::InvalidDraftTokens);
        }
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        let record = BundleDraftTokensV1 {
            struct_size: struct_size_v1::<BundleDraftTokensV1>(),
            reserved: 0,
            request_id,
            token_count,
            token_bytes: BundleByteViewV1::from_bytes(&bytes),
        };
        validate_draft_tokens_v1(record)?;
        trampoline::invoke_emit_draft_tokens_v1(&self.raw, record)
    }
}

/// Charges one row against a sink's declared limits before it is delivered.
pub(crate) fn charge_result_capacity_v1(
    limits: ResultSinkLimitsV1,
    emitted_rows: &mut u64,
    emitted_logits: &mut u64,
    logit_count: u64,
) -> Result<(), BundleAbiError> {
    charge_result_capacity_rows_v1(limits, emitted_rows, emitted_logits, 1, logit_count)
}

pub(crate) fn charge_result_capacity_rows_v1(
    limits: ResultSinkLimitsV1,
    emitted_rows: &mut u64,
    emitted_logits: &mut u64,
    row_count: u64,
    logit_count: u64,
) -> Result<(), BundleAbiError> {
    if row_count == 0 || logit_count == 0 || logit_count > limits.maximum_logits_per_row {
        return Err(BundleAbiError::ExecuteResultCapacity);
    }
    let rows = emitted_rows
        .checked_add(row_count)
        .ok_or(BundleAbiError::ExecuteResultCapacity)?;
    if rows > limits.maximum_rows {
        return Err(BundleAbiError::ExecuteResultCapacity);
    }
    let total = emitted_logits
        .checked_add(
            row_count
                .checked_mul(logit_count)
                .ok_or(BundleAbiError::ExecuteResultCapacity)?,
        )
        .ok_or(BundleAbiError::ExecuteResultCapacity)?;
    if total > limits.maximum_total_logits {
        return Err(BundleAbiError::ExecuteResultCapacity);
    }
    *emitted_rows = rows;
    *emitted_logits = total;
    Ok(())
}

pub type BundleCreateFnV1 =
    unsafe extern "C" fn(*const BundleHostServicesV1, BundleByteViewV1) -> BundleCreateResultV1;
pub type BundleExecuteFnV1 = unsafe extern "C" fn(
    BundleHandleV1,
    BundleByteViewV1,
    BundleExecuteResultSinkV1,
) -> BundleStatusV1;

// The three transaction callbacks carry no pointer or view. A caller cannot make
// them read memory it does not own, so they stay callable without unsafe. A V1
// implementation rejects a handle whose generation it did not issue.
pub type BundleCommitFnV1 = extern "C" fn(BundleHandleV1) -> BundleStatusV1;
pub type BundleCancelFnV1 = extern "C" fn(BundleHandleV1) -> BundleStatusV1;
pub type BundleDestroyFnV1 = extern "C" fn(BundleHandleV1) -> BundleStatusV1;

/// The fixed prefix the dynamic loader reads before copying a V1 API table.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleApiHeaderV1 {
    pub struct_size: u32,
    pub abi_version: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleRawApiV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub create: Option<BundleCreateFnV1>,
    pub execute: Option<BundleExecuteFnV1>,
    pub commit: Option<BundleCommitFnV1>,
    pub cancel: Option<BundleCancelFnV1>,
    pub destroy: Option<BundleDestroyFnV1>,
}

impl BundleRawApiV1 {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            abi_version: BUNDLE_ABI_VERSION_V1,
            create: None,
            execute: None,
            commit: None,
            cancel: None,
            destroy: None,
        }
    }

    #[must_use]
    pub const fn header(&self) -> BundleApiHeaderV1 {
        BundleApiHeaderV1 {
            struct_size: self.struct_size,
            abi_version: self.abi_version,
        }
    }
}

/// Capability requiring an advertised, non-null request retirement callback.
pub const REQUEST_RETIREMENT_CAPABILITY_V1: &str = "chop.request-retirement.v1";
/// The host returns scored prompt rows through the V2 row contract.
pub const PROMPT_ROWS_CAPABILITY_V2: &str = "chop.prompt-rows.v2";

pub const SERVING_KV_CAPABILITY_V1: &str = "chop.serving-kv.v1";
pub const SPECULATIVE_CAPABILITY_V1: &str = "chop.speculative.v1";
/// Capability a package requires when its bundle reads the served item count:
/// the `u32` live-sequence count that then closes the host's
/// `chopbundle-config-v1` create record. A host sends that field only to
/// packages that require this, so bundles built before the field existed,
/// which refuse any trailing byte, keep receiving the record they expect.
pub const SERVED_ITEMS_CAPABILITY_V1: &str = "chop.served-items.v1";
/// Packages with this capability receive a tagged model-options tail before
/// the optional tap producer: ASCII `chop.model-options.v1`, a little-endian
/// u32 JSON byte length, then a JSON object of resolved typed values in name
/// order. The object contains every declared option, including defaults.
/// Packages without the capability receive no extra bytes. Bundles must
/// validate names, types and compiled limits before allocating or capturing.
pub const MODEL_OPTIONS_CAPABILITY_V1: &str = "chop.model-options.v1";
pub const MODEL_OPTIONS_TAG_V1: &[u8] = b"chop.model-options.v1";

/// The create record appends the served context limit as u64 after its item count.
/// Requires served-items and batch limits; older packages receive no extra bytes.
pub const SERVED_CONTEXT_CAPABILITY_V1: &str = "chop.served-context.v1";
/// Capability a package requires when its declared `state_layout` bytes are
/// per state slot rather than for the whole instance. The host then declares,
/// sends in the create record and allocates `bytes * raw_state_slots_v1(served)`
/// per rank, where `served` is the served item count the record closes with;
/// the package must also require `chop.served-items.v1` and declare batch
/// limits. Other packages keep their fixed state size.
pub const STATE_PER_ITEM_CAPABILITY_V1: &str = "chop.state-per-item.v1";

/// State slots of an instance serving `served` items under
/// `chop.state-per-item.v1`: the count rounded up to a power of two, the
/// largest graph bucket, which per-slot kernels are built for.
#[must_use]
pub const fn raw_state_slots_v1(served: u32) -> u64 {
    (served as u64).next_power_of_two()
}

/// Capability for packages whose compiled images place views in host memory.
///
/// A manifest view marked `"placement": "host"` is not uploaded to the device.
/// The host reads its bytes into page-locked, device-mapped host memory, and
/// `resolve_compiled_image_view` answers that view with a buffer the device
/// reads over PCIe. The rank arena holds the rest of the rank image: each host
/// span, rounded up to the image alignment, is cut out and the bytes after it
/// move down by that length. The whole-arena view is therefore shorter than the
/// rank image by the rounded host bytes. Such an image aligns no view above
/// the image alignment and leaves every rank at least one device byte. A
/// package declaring host views must require this capability, so a host
/// without it refuses the package.
pub const HOST_RESIDENT_VIEWS_CAPABILITY_V1: &str = "chop.host-resident-views.v1";

pub type BundleRetireRequestFnV1 = extern "C" fn(BundleHandleV1, u64) -> BundleStatusV1;

/// Optional V1 extension. The entry point still returns a pointer to `base`.
/// Retirement is synchronous, after commit/cancel and before the next execute.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleApiWithRequestRetirementV1 {
    pub base: BundleRawApiV1,
    pub retire_request_v1: Option<BundleRetireRequestFnV1>,
}

pub(crate) const REQUEST_RETIREMENT_END_V1: usize =
    std::mem::offset_of!(BundleApiWithRequestRetirementV1, retire_request_v1)
        + std::mem::size_of::<Option<BundleRetireRequestFnV1>>();

/// One array element passed to `commit_accepted_v1`; the size is exact.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleKeptRowsV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub request_id: u64,
    pub kept_rows: u32,
    pub reserved_tail: u32,
}

pub type BundleCommitAcceptedFnV1 =
    unsafe extern "C" fn(BundleHandleV1, *const BundleKeptRowsV1, u32) -> BundleStatusV1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleApiWithSpeculationV1 {
    pub base: BundleApiWithRequestRetirementV1,
    pub commit_accepted_v1: Option<BundleCommitAcceptedFnV1>,
}

pub(crate) const COMMIT_ACCEPTED_END_V1: usize =
    std::mem::offset_of!(BundleApiWithSpeculationV1, commit_accepted_v1)
        + std::mem::size_of::<Option<BundleCommitAcceptedFnV1>>();

/// Capability for bundles whose manifest declares per-request state layouts.
/// Every batch then carries one host-owned state slot ID per item.
pub const RECURRENT_STATE_CAPABILITY_V1: &str = "chop.recurrent-state.v1";

/// Capability for bundles that allocate storage for every declared
/// request-state slot during create, before they acquire the serving KV pool.
/// That storage is already outside the free memory the pool is sized from, so
/// the host reserves nothing more for it.
pub const RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1: &str = "chop.recurrent-state.preallocated.v1";

/// Capability requiring the prefix-state callbacks of the API tail below.
pub const PREFIX_CACHE_CAPABILITY_V1: &str = "chop.prefix-cache.v1";

/// Capability extending prefix capture to committed prefill, decode and
/// verify boundaries. Restore followed by the next decode or verify item
/// reproduces the uncaptured rows under the bundle's numerical contract.
/// A successful capture completes every preceding device write to the
/// request's KV pages and snapshot pages on every rank before returning.
/// Refused captures leave request state unchanged; unconfirmed completion
/// is fatal. Requires `PREFIX_CACHE_CAPABILITY_V1`, with no new ABI tail.
pub const KV_OFFLOAD_CAPABILITY_V1: &str = "chop.kv-offload.v1";

/// Capability for bundles that read explicit per-row 3-axis RoPE positions.
/// A batch carrying any such positions then arrives as `chopbatch-v5`.
pub const POSITION_AXES_CAPABILITY_V1: &str = "chop.position-axes.v1";

/// Capability of bundles whose serving KV pages store K and V as FP8 E4M3.
/// The host reports the pool's tokens as FP8 KV tokens.
pub const FP8_KV_CAPABILITY_V1: &str = "chop.kv-cache.fp8-e4m3.v1";

/// How many serving-pool pages one bundle snapshot needs; may be zero.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundlePrefixStateInfoV1 {
    pub struct_size: u32,
    pub status: BundleStatusV1,
    pub state_pages: u64,
}

impl BundlePrefixStateInfoV1 {
    #[must_use]
    pub const fn success(state_pages: u64) -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            status: BundleStatusV1::Success,
            state_pages,
        }
    }

    #[must_use]
    pub const fn failure(status: BundleStatusV1) -> Self {
        Self {
            struct_size: struct_size_v1::<Self>(),
            status,
            state_pages: 0,
        }
    }
}

/// One capture or restore of a request's bundle-owned state at `tokens`.
///
/// `token_ids` holds `tokens` little-endian `u32` IDs for a restore and is
/// empty for a capture. `pages` holds the snapshot's little-endian `u64`
/// serving-pool page IDs.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundlePrefixStateV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub request_id: u64,
    pub tokens: u64,
    pub token_ids: BundleByteViewV1,
    pub pages: BundleByteViewV1,
}

pub type BundleDescribePrefixStateFnV1 = extern "C" fn(BundleHandleV1) -> BundlePrefixStateInfoV1;
pub type BundlePrefixStateFnV1 =
    unsafe extern "C" fn(BundleHandleV1, BundlePrefixStateV1) -> BundleStatusV1;

/// The speculation table plus the prefix-state callbacks. The entry still
/// returns a pointer to the innermost base; `commit_accepted_v1` may be null.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleApiWithPrefixCacheV1 {
    pub base: BundleApiWithSpeculationV1,
    pub describe_prefix_state_v1: Option<BundleDescribePrefixStateFnV1>,
    pub capture_prefix_state_v1: Option<BundlePrefixStateFnV1>,
    pub restore_prefix_state_v1: Option<BundlePrefixStateFnV1>,
}

pub(crate) const PREFIX_CACHE_END_V1: usize =
    std::mem::offset_of!(BundleApiWithPrefixCacheV1, restore_prefix_state_v1)
        + std::mem::size_of::<Option<BundlePrefixStateFnV1>>();

/// Capability requiring the state-slot restore of the API tail below.
pub const PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1: &str = "chop.prefix-cache.state-slot.v1";

/// A restore that also names the state slot the host assigned the request.
pub type BundleRestorePrefixStateInSlotFnV1 =
    unsafe extern "C" fn(BundleHandleV1, BundlePrefixStateV1, u64) -> BundleStatusV1;

/// The prefix-cache table plus the state-slot restore. The entry still
/// returns a pointer to the innermost base.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleApiWithPrefixStateSlotV1 {
    pub base: BundleApiWithPrefixCacheV1,
    pub restore_prefix_state_in_slot_v1: Option<BundleRestorePrefixStateInSlotFnV1>,
}

pub(crate) const PREFIX_STATE_SLOT_END_V1: usize =
    std::mem::offset_of!(
        BundleApiWithPrefixStateSlotV1,
        restore_prefix_state_in_slot_v1
    ) + std::mem::size_of::<Option<BundleRestorePrefixStateInSlotFnV1>>();

/// The three prefix-state callbacks, present together or not at all.
#[derive(Clone, Copy)]
pub(crate) struct PrefixCacheCallbacksV1 {
    describe: BundleDescribePrefixStateFnV1,
    capture: BundlePrefixStateFnV1,
    restore: BundlePrefixStateFnV1,
}

impl PrefixCacheCallbacksV1 {
    pub(crate) fn from_table(
        describe: Option<BundleDescribePrefixStateFnV1>,
        capture: Option<BundlePrefixStateFnV1>,
        restore: Option<BundlePrefixStateFnV1>,
    ) -> Option<Self> {
        Some(Self {
            describe: describe?,
            capture: capture?,
            restore: restore?,
        })
    }
}

pub type BundleGetApiV1Fn = unsafe extern "C" fn() -> *const BundleRawApiV1;

/// A weight plan sink callback: the encoded plan, or refusal text.
pub type BundleWeightPlanEmitFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleByteViewV1) -> BundleStatusV1;
/// `chop_bundle_weight_plan_v1(request, sink)`; see [`weight_plan`].
pub type BundleWeightPlanFnV1 =
    unsafe extern "C" fn(BundleByteViewV1, weight_plan::BundleWeightPlanSinkV1) -> BundleStatusV1;
/// `chop_bundle_weight_transform_v1(call)`; see [`weight_plan`].
pub type BundleWeightTransformFnV1 =
    unsafe extern "C" fn(*const weight_plan::BundleWeightTransformCallV1) -> BundleStatusV1;

/// A frontend sink callback: the answer document, or a refusal document.
pub type BundleFrontendEmitFnV1 =
    unsafe extern "C" fn(*mut c_void, BundleByteViewV1) -> BundleStatusV1;
/// `open(component, sink, frontend)`; see [`frontend`].
pub type BundleFrontendOpenFnV1 = unsafe extern "C" fn(
    BundleByteViewV1,
    frontend::BundleFrontendSinkV1,
    *mut *mut frontend::BundleFrontendV1,
) -> BundleStatusV1;
/// `close(frontend)`.
pub type BundleFrontendCloseFnV1 =
    unsafe extern "C" fn(*mut frontend::BundleFrontendV1) -> BundleStatusV1;
/// `render` and `constraint`: `(frontend, request, sink)`.
pub type BundleFrontendDocumentFnV1 = unsafe extern "C" fn(
    *const frontend::BundleFrontendV1,
    BundleByteViewV1,
    frontend::BundleFrontendSinkV1,
) -> BundleStatusV1;
/// `parser_open(frontend, request, sink, parser)`.
pub type BundleFrontendParserOpenFnV1 = unsafe extern "C" fn(
    *const frontend::BundleFrontendV1,
    BundleByteViewV1,
    frontend::BundleFrontendSinkV1,
    *mut *mut frontend::BundleFrontendParserV1,
) -> BundleStatusV1;
/// `parser_push(parser, text, flags, sink)`.
pub type BundleFrontendParserPushFnV1 = unsafe extern "C" fn(
    *mut frontend::BundleFrontendParserV1,
    BundleByteViewV1,
    u32,
    frontend::BundleFrontendSinkV1,
) -> BundleStatusV1;
/// `parser_close(parser)`.
pub type BundleFrontendParserCloseFnV1 =
    unsafe extern "C" fn(*mut frontend::BundleFrontendParserV1) -> BundleStatusV1;
/// `chop_bundle_frontend_api_v1()`, which returns the bundle's static table.
pub type BundleFrontendApiFnV1 = unsafe extern "C" fn() -> *const frontend::BundleFrontendTableV1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BundleAbiError {
    AbiVersion {
        surface: &'static str,
        actual: u32,
    },
    StructSize {
        surface: &'static str,
        expected: u32,
        actual: u32,
    },
    /// A record whose declared length must be V1's exactly, not merely at least.
    ///
    /// This is the array-element rule: the callback carries no stride, so a
    /// caller declaring any other length describes an array this reader would
    /// step through wrongly.
    StructSizeExact {
        surface: &'static str,
        expected: u32,
        actual: u32,
    },
    MissingApiCallback(&'static str),
    MissingHostService(&'static str),
    MissingExecuteResultCallback,
    InvalidByteView,
    InvalidMutableByteView,
    InvalidReserved {
        surface: &'static str,
        actual: u32,
    },
    InvalidAlignment {
        surface: &'static str,
        actual: u64,
    },
    InvalidRange(&'static str),
    InvalidKernelArgumentKind(u32),
    /// A kernel argument whose kind selects a payload it did not carry.
    MissingKernelArgumentPayload(&'static str),
    /// A kernel argument carrying a payload its kind does not select.
    UnselectedKernelArgumentPayload(&'static str),
    /// A by-value device-pointer array whose slot count or slots are invalid.
    InvalidDevicePointerArraySlots(u32),
    /// A custom operand set a flag bit ABI V1 does not define.
    InvalidCustomOperandFlags(u32),
    /// A lent launch named an address space outside the closed V1 set.
    InvalidCustomAddressSpace(u32),
    /// A lent launch record contradicts itself, naming the field that does.
    InvalidCustomLaunch(&'static str),
    /// A custom operation asked for a completion policy V1 does not define.
    InvalidCustomCompletion(u32),
    /// A host reported a bundle failure without the operation's own code.
    MissingCustomOperationStatus,
    /// A custom operation the caller did not supply, or a null callback.
    MissingCustomOperationCallback,
    /// The nonzero code one custom operation's own callback returned.
    CustomOperation(i32),
    InvalidTensorMapElement(u32),
    InvalidTensorMapSwizzle(u32),
    /// One tensor-map extent, stride, or box dimension the encoding refuses.
    InvalidTensorMapGeometry(&'static str),
    InvalidLaunchFlags(u32),
    /// A thread-block cluster no launch could take, naming the rule it breaks.
    InvalidLaunchCluster(&'static str),
    LengthOutOfRange {
        surface: &'static str,
        actual: u64,
    },
    InvalidPointerLength(&'static str),
    NullPointer(&'static str),
    MisalignedPointer(&'static str),
    InvalidLogitsRow,
    InvalidLogitsView,
    InvalidDraftTokens,
    InvalidKeptRows,
    ExecuteResultCapacity,
    InvalidHandle,
    Status(BundleStatusV1),
    Lifecycle(&'static str),
    Poisoned,
}

impl fmt::Display for BundleAbiError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AbiVersion { surface, actual } => write!(
                formatter,
                "{surface} ABI version {actual} does not match V1"
            ),
            Self::StructSize {
                surface,
                expected,
                actual,
            } => write!(
                formatter,
                "{surface} struct size {actual} is smaller than required V1 size {expected}"
            ),
            Self::StructSizeExact {
                surface,
                expected,
                actual,
            } => write!(
                formatter,
                "{surface} struct size {actual} is not the exact V1 size {expected}, which an \
                 array element with no carried stride must be"
            ),
            Self::MissingApiCallback(callback) => {
                write!(formatter, "bundle API callback '{callback}' is missing")
            }
            Self::MissingHostService(service) => {
                write!(formatter, "bundle host service '{service}' is missing")
            }
            Self::MissingExecuteResultCallback => {
                formatter.write_str("execute result sink callback is missing")
            }
            Self::InvalidByteView => {
                formatter.write_str("byte view has an invalid pointer-length pair")
            }
            Self::InvalidMutableByteView => {
                formatter.write_str("mutable byte view has an invalid pointer-length pair")
            }
            Self::InvalidReserved { surface, actual } => {
                write!(
                    formatter,
                    "{surface} reserved field must be zero, got {actual}"
                )
            }
            Self::InvalidAlignment { surface, actual } => {
                write!(
                    formatter,
                    "{surface} alignment {actual} is not a nonzero power of two"
                )
            }
            Self::InvalidRange(surface) => write!(formatter, "{surface} range is invalid"),
            Self::InvalidKernelArgumentKind(kind) => {
                write!(formatter, "kernel argument kind {kind} is invalid")
            }
            Self::MissingKernelArgumentPayload(surface) => write!(
                formatter,
                "kernel argument selects a {surface} payload it did not carry"
            ),
            Self::UnselectedKernelArgumentPayload(surface) => write!(
                formatter,
                "kernel argument carries a {surface} payload its kind does not select"
            ),
            Self::InvalidCustomOperandFlags(flags) => write!(
                formatter,
                "custom operand flags {flags:#010x} set a bit ABI V1 does not define"
            ),
            Self::InvalidCustomAddressSpace(address_space) => write!(
                formatter,
                "custom launch address space {address_space} is outside the closed V1 set"
            ),
            Self::InvalidCustomLaunch(surface) => {
                write!(formatter, "custom launch {surface} is invalid")
            }
            Self::InvalidCustomCompletion(flags) => write!(
                formatter,
                "custom operation flags {flags:#010x} set a bit ABI V1 does not define"
            ),
            Self::MissingCustomOperationStatus => formatter
                .write_str("the host reported a bundle failure without the operation's own status"),
            Self::MissingCustomOperationCallback => {
                formatter.write_str("custom operation callback is missing")
            }
            Self::CustomOperation(code) => {
                write!(formatter, "custom operation failed with code {code}")
            }
            Self::InvalidDevicePointerArraySlots(slot_count) => write!(
                formatter,
                "device pointer array declares {slot_count} slots, which is outside 1 through \
                 {BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS}, or a slot inside that count \
                 is neither a valid span nor a null pointer"
            ),
            Self::InvalidTensorMapElement(element_type) => write!(
                formatter,
                "tensor map element type {element_type} is outside the closed V1 set"
            ),
            Self::InvalidTensorMapSwizzle(swizzle) => write!(
                formatter,
                "tensor map swizzle {swizzle} is outside the closed V1 set"
            ),
            Self::InvalidTensorMapGeometry(surface) => {
                write!(formatter, "tensor map {surface} is invalid")
            }
            Self::InvalidLaunchFlags(flags) => write!(
                formatter,
                "launch config flags {flags:#010x} set a bit ABI V1 does not define"
            ),
            Self::InvalidLaunchCluster(surface) => {
                write!(formatter, "launch config {surface} is invalid")
            }
            Self::LengthOutOfRange { surface, actual } => {
                write!(
                    formatter,
                    "{surface} length {actual} cannot form a Rust slice"
                )
            }
            Self::InvalidPointerLength(surface) => {
                write!(formatter, "{surface} has an invalid pointer-length pair")
            }
            Self::NullPointer(surface) => write!(formatter, "{surface} pointer is null"),
            Self::MisalignedPointer(surface) => {
                write!(formatter, "{surface} pointer is misaligned")
            }
            Self::InvalidLogitsRow => formatter.write_str("logits row is invalid"),
            Self::InvalidLogitsView => formatter.write_str("device logits view is invalid"),
            Self::InvalidDraftTokens => formatter.write_str("draft token record is invalid"),
            Self::InvalidKeptRows => formatter.write_str("kept-row record is invalid"),
            Self::ExecuteResultCapacity => {
                formatter.write_str("execute result exceeds the declared sink capacity")
            }
            Self::InvalidHandle => formatter.write_str("bundle handle has no generation or slot"),
            Self::Status(status) => match status.name() {
                Some(name) => write!(formatter, "bundle ABI status {} ({name})", status.0),
                None => write!(formatter, "bundle ABI status {}", status.0),
            },
            Self::Lifecycle(state) => write!(formatter, "bundle lifecycle does not allow {state}"),
            Self::Poisoned => {
                formatter.write_str("bundle instance is poisoned after a callback error")
            }
        }
    }
}

impl std::error::Error for BundleAbiError {}

pub fn validate_byte_view_v1(view: BundleByteViewV1) -> Result<(), BundleAbiError> {
    if view.data.is_null() == (view.length == 0) {
        Ok(())
    } else {
        Err(BundleAbiError::InvalidByteView)
    }
}

pub fn validate_mut_byte_view_v1(view: BundleMutByteViewV1) -> Result<(), BundleAbiError> {
    if view.data.is_null() == (view.length == 0) {
        Ok(())
    } else {
        Err(BundleAbiError::InvalidMutableByteView)
    }
}

/// Validates a device-buffer range before a host service resolves it.
///
/// Its declared length is exact rather than a range. This record is embedded by
/// value in five others, and each parent computes the offset of every field
/// after it from this length, so a tail here would move the parent's own fields
/// rather than only its own. It is also passed on its own to several services,
/// where the call already fixes its layout; one rule covers both, because a
/// length this ABI could not accept nested is not one it accepts anywhere.
pub fn validate_device_buffer_view_v1(
    view: BundleDeviceBufferViewV1,
) -> Result<(), BundleAbiError> {
    validate_device_buffer_view_struct_size_v1(view.struct_size)?;
    validate_reserved("device buffer view", view.reserved)?;
    view.buffer.validate()?;
    validate_range("device buffer view", view.offset, view.length)
}

/// The exact-length rule for one device-buffer view, a by-value nested record.
pub fn validate_device_buffer_view_struct_size_v1(struct_size: u32) -> Result<(), BundleAbiError> {
    validate_exact_struct_size(
        "device buffer view",
        struct_size,
        struct_size_v1::<BundleDeviceBufferViewV1>(),
    )
}

/// The exact-length rule for one physical KV layout, a by-value nested record.
pub fn validate_physical_kv_layout_struct_size_v1(struct_size: u32) -> Result<(), BundleAbiError> {
    validate_exact_struct_size(
        "physical KV layout",
        struct_size,
        struct_size_v1::<BundlePhysicalKvLayoutV1>(),
    )
}

/// Validates a compiled-image view and its nested device-buffer range.
pub fn validate_compiled_image_view_v1(
    view: BundleRawCompiledImageViewV1,
) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "compiled image view",
        view.struct_size,
        struct_size_v1::<BundleRawCompiledImageViewV1>(),
    )?;
    validate_device_buffer_view_v1(view.buffer)?;
    validate_alignment("compiled image view", view.alignment)
}

/// Validates fixed authenticated-component metadata before allocating its buffers.
pub fn validate_authenticated_component_metadata_v1(
    metadata: BundleAuthenticatedComponentMetadataV1,
) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "authenticated component metadata",
        metadata.struct_size,
        struct_size_v1::<BundleAuthenticatedComponentMetadataV1>(),
    )?;
    if metadata.reserved != 0 {
        return Err(BundleAbiError::InvalidReserved {
            surface: "authenticated component metadata",
            actual: metadata.reserved,
        });
    }
    if metadata.role_length == 0
        || metadata.role_length > BUNDLE_AUTHENTICATED_COMPONENT_ROLE_BYTES_MAX_V1
    {
        return Err(BundleAbiError::LengthOutOfRange {
            surface: "authenticated component role",
            actual: metadata.role_length,
        });
    }
    if metadata.byte_length > BUNDLE_AUTHENTICATED_COMPONENT_BYTES_MAX_V1 {
        return Err(BundleAbiError::LengthOutOfRange {
            surface: "authenticated component bytes",
            actual: metadata.byte_length,
        });
    }
    if !metadata
        .sha256
        .iter()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte))
    {
        return Err(BundleAbiError::InvalidRange(
            "authenticated component SHA-256",
        ));
    }
    Ok(())
}

/// Validates a generic CUDA launch shape without applying model-specific limits.
pub fn validate_launch_config_v1(config: BundleLaunchConfigV1) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "launch config",
        config.struct_size,
        struct_size_v1::<BundleLaunchConfigV1>(),
    )?;
    // A host that does not know a flag cannot apply it, so an undefined bit is
    // refused here rather than launched without whatever it asked for.
    if config.flags & !BUNDLE_LAUNCH_FLAGS_V1_KNOWN != 0 {
        return Err(BundleAbiError::InvalidLaunchFlags(config.flags));
    }
    for (surface, value) in [
        ("launch config grid_x", config.grid_x),
        ("launch config grid_y", config.grid_y),
        ("launch config grid_z", config.grid_z),
        ("launch config block_x", config.block_x),
        ("launch config block_y", config.block_y),
        ("launch config block_z", config.block_z),
    ] {
        if value == 0 {
            return Err(BundleAbiError::InvalidRange(surface));
        }
    }
    validate_launch_cluster_v1(config)
}

/// Validates the cluster one launch asks its blocks to be grouped into.
///
/// One block in every dimension is the default and asks for nothing. Any other
/// shape has to be one CUDA can form: every extent at least one block, the
/// blocks inside the portable bound, and a grid that divides into whole
/// clusters. A shape failing one of those is refused here, so no host has to
/// decide whether to round it to one that would have launched.
fn validate_launch_cluster_v1(config: BundleLaunchConfigV1) -> Result<(), BundleAbiError> {
    let cluster = [config.cluster_x, config.cluster_y, config.cluster_z];
    if cluster.contains(&0) {
        return Err(BundleAbiError::InvalidLaunchCluster("cluster dimension"));
    }
    let blocks = cluster
        .iter()
        .try_fold(1_u32, |blocks, extent| blocks.checked_mul(*extent))
        .ok_or(BundleAbiError::InvalidLaunchCluster("cluster blocks"))?;
    if blocks > BUNDLE_LAUNCH_V1_MAX_CLUSTER_BLOCKS {
        return Err(BundleAbiError::InvalidLaunchCluster("cluster blocks"));
    }
    let grid = [config.grid_x, config.grid_y, config.grid_z];
    for (extent, grid_extent) in cluster.into_iter().zip(grid) {
        if !grid_extent.is_multiple_of(extent) {
            return Err(BundleAbiError::InvalidLaunchCluster("cluster grid"));
        }
    }
    Ok(())
}

/// Validates one kernel argument's kind and the payloads it does or does not
/// select.
///
/// Everything decided here is decided without reading through a pointer: the
/// record's own length, its kind, and whether each pointer-bearing field agrees
/// with that kind. The payload a kind selects must be present and the ones it
/// does not select must be absent, so a record that names two payloads is
/// refused before either could be read as the one that mattered.
///
/// Reading a selected payload is the boundary's own work, and validating what
/// it read is [`validate_tensor_map_v1`] or
/// [`validate_device_pointer_array_v1`], exactly as it was when this record
/// carried them by value.
pub fn validate_kernel_argument_v1(argument: BundleKernelArgumentV1) -> Result<(), BundleAbiError> {
    validate_kernel_argument_struct_size_v1(argument.struct_size)?;
    let selected = match argument.kind {
        BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_BUFFER => {
            validate_device_buffer_view_v1(argument.buffer)?;
            KernelArgumentPayloadV1::None
        }
        BUNDLE_KERNEL_ARGUMENT_V1_U64
        | BUNDLE_KERNEL_ARGUMENT_V1_U32
        | BUNDLE_KERNEL_ARGUMENT_V1_I32
        | BUNDLE_KERNEL_ARGUMENT_V1_NULL_DEVICE_POINTER => KernelArgumentPayloadV1::None,
        BUNDLE_KERNEL_ARGUMENT_V1_OPAQUE_BYTES => {
            validate_byte_view_v1(argument.opaque)?;
            KernelArgumentPayloadV1::Opaque
        }
        BUNDLE_KERNEL_ARGUMENT_V1_TENSOR_MAP => KernelArgumentPayloadV1::TensorMap,
        BUNDLE_KERNEL_ARGUMENT_V1_DEVICE_POINTER_ARRAY => KernelArgumentPayloadV1::PointerArray,
        kind => return Err(BundleAbiError::InvalidKernelArgumentKind(kind)),
    };
    validate_kernel_argument_payloads_v1(argument, selected)
}

/// Checks a record whose declared length must be exactly V1's.
///
/// Most V1 records are read at their own layout out of a caller's storage, so a
/// longer one is accepted and its tail left unread. That works only where the
/// record's own start is the thing a reader was handed. Two kinds of record are
/// not read that way, and neither admits a tail in either direction:
///
/// An **array element** is reached by advancing from the element before it, and
/// none of the three callbacks that take an array carries an element stride. A
/// caller whose record is longer has a longer stride, so the second element this
/// reader looks at would start inside the first; a shorter one runs the array
/// off its own end. [`validate_kernel_argument_struct_size_v1`] and
/// [`validate_custom_operand_struct_size_v1`] are these.
///
/// A **by-value nested record** is reached at a fixed offset inside its parent,
/// and every parent field after it sits at an offset computed from its length. A
/// caller whose nested record is longer has shifted all of them, so the parent's
/// own later fields would be read from bytes that mean something else, and the
/// parent's `>=` rule would accept it.
/// [`validate_device_buffer_view_v1`] and [`validate_physical_kv_layout_v1`] are
/// these, and they are checked before any later parent field is used.
///
/// A revision that wants a longer one of either kind is changing where every
/// reader of a container or an array looks, which is a new ABI version rather
/// than an addition to this one.
fn validate_exact_struct_size(
    surface: &'static str,
    actual: u32,
    expected: u32,
) -> Result<(), BundleAbiError> {
    if actual == expected {
        Ok(())
    } else {
        Err(BundleAbiError::StructSizeExact {
            surface,
            expected,
            actual,
        })
    }
}

/// The exact-length rule for one kernel argument, an element of the array
/// `launch_cuda_symbol` and `execute_collective` take.
pub fn validate_kernel_argument_struct_size_v1(struct_size: u32) -> Result<(), BundleAbiError> {
    validate_exact_struct_size(
        "kernel argument",
        struct_size,
        struct_size_v1::<BundleKernelArgumentV1>(),
    )
}

/// The exact-length rule for one custom operand, an element of the array
/// `run_custom_operation` takes.
pub fn validate_custom_operand_struct_size_v1(struct_size: u32) -> Result<(), BundleAbiError> {
    validate_exact_struct_size(
        "custom operand",
        struct_size,
        struct_size_v1::<BundleCustomOperandV1>(),
    )
}

/// Which payload one kernel argument kind selects, if it selects any.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum KernelArgumentPayloadV1 {
    None,
    Opaque,
    TensorMap,
    PointerArray,
}

/// Requires the selected payload and refuses every unselected one.
fn validate_kernel_argument_payloads_v1(
    argument: BundleKernelArgumentV1,
    selected: KernelArgumentPayloadV1,
) -> Result<(), BundleAbiError> {
    for (payload, surface, present) in [
        (
            KernelArgumentPayloadV1::TensorMap,
            "tensor map",
            !argument.tensor_map.is_null(),
        ),
        (
            KernelArgumentPayloadV1::PointerArray,
            "device pointer array",
            !argument.pointer_array.is_null(),
        ),
        (
            KernelArgumentPayloadV1::Opaque,
            "opaque bytes",
            !argument.opaque.data.is_null() || argument.opaque.length != 0,
        ),
    ] {
        if payload == selected {
            if !present {
                return Err(BundleAbiError::MissingKernelArgumentPayload(surface));
            }
        } else if present {
            return Err(BundleAbiError::UnselectedKernelArgumentPayload(surface));
        }
    }
    Ok(())
}

/// Validates one by-value device-pointer array before any host resolves it.
///
/// The slot count is what makes this bounded, so it is checked first and every
/// slot inside it is checked as a device-buffer view would be. A null slot is
/// the null pointer and carries nothing else: a view that names the null handle
/// with a nonzero offset or length is a slot the caller filled in wrongly, not
/// an unused one, and it is refused rather than silently launched as null. The
/// slots past the count are not read at all, so nothing there can decide
/// anything.
pub fn validate_device_pointer_array_v1(
    array: BundleDevicePointerArrayV1,
) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "device pointer array",
        array.struct_size,
        struct_size_v1::<BundleDevicePointerArrayV1>(),
    )?;
    if array.slot_count == 0 || array.slot_count > BUNDLE_DEVICE_POINTER_ARRAY_V1_MAX_SLOTS {
        return Err(BundleAbiError::InvalidDevicePointerArraySlots(
            array.slot_count,
        ));
    }
    for slot in &array.slots[..array.slot_count as usize] {
        if slot.buffer.raw == 0 {
            // A null slot is checked at the same exact length as any other: the
            // slots are one inline array, so a slot declaring a different length
            // has moved every slot after it.
            validate_device_buffer_view_struct_size_v1(slot.struct_size)?;
            validate_reserved("device pointer array slot", slot.reserved)?;
            if slot.offset != 0 || slot.length != 0 {
                return Err(BundleAbiError::InvalidDevicePointerArraySlots(
                    array.slot_count,
                ));
            }
        } else {
            validate_device_buffer_view_v1(*slot)?;
        }
    }
    Ok(())
}

/// Validates one tiled tensor map before a host resolves its device address.
pub fn validate_tensor_map_v1(map: BundleTensorMapV1) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "tensor map",
        map.struct_size,
        struct_size_v1::<BundleTensorMapV1>(),
    )?;
    validate_reserved("tensor map", map.reserved)?;
    validate_device_buffer_view_v1(map.buffer)?;
    let element_bytes = tensor_map_element_bytes_v1(map.element_type)
        .ok_or(BundleAbiError::InvalidTensorMapElement(map.element_type))?;
    let swizzle_bytes = tensor_map_swizzle_bytes_v1(map.swizzle)
        .ok_or(BundleAbiError::InvalidTensorMapSwizzle(map.swizzle))?;
    let rank = usize::try_from(map.rank).unwrap_or(0);
    if !(1..=5).contains(&rank) {
        return Err(BundleAbiError::InvalidTensorMapGeometry("rank"));
    }
    if map.interleave > BUNDLE_TENSOR_MAP_INTERLEAVE_V1_BYTES_32
        || map.l2_promotion > BUNDLE_TENSOR_MAP_L2_PROMOTION_V1_BYTES_256
        || map.oob_fill > BUNDLE_TENSOR_MAP_OOB_FILL_V1_NAN_REQUEST_ZERO_FMA
    {
        return Err(BundleAbiError::InvalidTensorMapGeometry("enum"));
    }
    if (map.interleave != 0 && rank < 3)
        || (map.interleave == BUNDLE_TENSOR_MAP_INTERLEAVE_V1_BYTES_32
            && map.swizzle != BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_32)
        || (map.element_type == BUNDLE_TENSOR_MAP_ELEMENT_V1_16U6_ALIGN16B && map.interleave != 0)
    {
        return Err(BundleAbiError::InvalidTensorMapGeometry("interleave"));
    }
    let packed_16 = matches!(
        map.element_type,
        BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN16B | BUNDLE_TENSOR_MAP_ELEMENT_V1_16U6_ALIGN16B
    );
    if packed_16
        && !matches!(
            map.swizzle,
            BUNDLE_TENSOR_MAP_SWIZZLE_V1_NONE | BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_128
        )
    {
        return Err(BundleAbiError::InvalidTensorMapGeometry("packed swizzle"));
    }
    if map.oob_fill != 0
        && !matches!(
            map.element_type,
            BUNDLE_TENSOR_MAP_ELEMENT_V1_F16
                | BUNDLE_TENSOR_MAP_ELEMENT_V1_BF16
                | BUNDLE_TENSOR_MAP_ELEMENT_V1_F32
        )
    {
        return Err(BundleAbiError::InvalidTensorMapGeometry(
            "out-of-bounds fill",
        ));
    }
    for index in 0..5 {
        if index < rank {
            if map.global_dims[index] == 0 || map.global_dims[index] > (1_u64 << 32) {
                return Err(BundleAbiError::InvalidTensorMapGeometry("global dimension"));
            }
            if map.box_dims[index] == 0 || map.box_dims[index] > BUNDLE_TENSOR_MAP_MAX_BOX_V1 {
                return Err(BundleAbiError::InvalidTensorMapGeometry("box dimension"));
            }
            if !(1..=8).contains(&map.element_strides[index]) {
                return Err(BundleAbiError::InvalidTensorMapGeometry("element stride"));
            }
        } else if map.global_dims[index] != 0
            || map.box_dims[index] != 0
            || map.element_strides[index] != 0
        {
            return Err(BundleAbiError::InvalidTensorMapGeometry("unused dimension"));
        }
    }
    if (packed_16 && (map.global_dims[0] % 128 != 0 || map.box_dims[0] != 128))
        || (map.element_type == BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN8B
            && map.global_dims[0] % 2 != 0)
    {
        return Err(BundleAbiError::InvalidTensorMapGeometry("packed dimension"));
    }
    let inner_bytes = |elements: u64| -> Result<u64, BundleAbiError> {
        if map.element_type == BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN8B {
            elements
                .checked_add(1)
                .map(|value| value / 2)
                .ok_or(BundleAbiError::InvalidTensorMapGeometry("row width"))
        } else {
            elements
                .checked_mul(element_bytes)
                .ok_or(BundleAbiError::InvalidTensorMapGeometry("row width"))
        }
    };
    let row_bytes = inner_bytes(map.global_dims[0])?;
    let box_bytes = inner_bytes(u64::from(map.box_dims[0]))?;
    if (map.interleave == 0 && box_bytes % 16 != 0)
        || (map.interleave == 0 && swizzle_bytes != 0 && box_bytes > swizzle_bytes)
    {
        return Err(BundleAbiError::InvalidTensorMapGeometry("box width"));
    }
    let alignment = if packed_16 || map.interleave == BUNDLE_TENSOR_MAP_INTERLEAVE_V1_BYTES_32 {
        32
    } else {
        BUNDLE_TENSOR_MAP_ADDRESS_ALIGNMENT_V1
    };
    if map.buffer.offset % alignment != 0 {
        return Err(BundleAbiError::InvalidTensorMapGeometry("base alignment"));
    }
    let mut footprint = row_bytes;
    for index in 0..4 {
        let stride = map.global_strides_bytes[index];
        if index < rank - 1 {
            if stride == 0 || stride >= (1_u64 << 40) || stride % alignment != 0 {
                return Err(BundleAbiError::InvalidTensorMapGeometry("stride"));
            }
            let previous_extent = if index == 0 {
                row_bytes
            } else {
                map.global_strides_bytes[index - 1]
                    .checked_mul(map.global_dims[index])
                    .ok_or(BundleAbiError::InvalidTensorMapGeometry("footprint"))?
            };
            if stride < previous_extent {
                return Err(BundleAbiError::InvalidTensorMapGeometry(if index == 0 {
                    "row width"
                } else {
                    "stride"
                }));
            }
            footprint = footprint
                .checked_add(
                    stride
                        .checked_mul(map.global_dims[index + 1] - 1)
                        .ok_or(BundleAbiError::InvalidTensorMapGeometry("footprint"))?,
                )
                .ok_or(BundleAbiError::InvalidTensorMapGeometry("footprint"))?;
        } else if stride != 0 {
            return Err(BundleAbiError::InvalidTensorMapGeometry("unused stride"));
        }
    }
    if footprint > map.buffer.length {
        return Err(BundleAbiError::InvalidTensorMapGeometry("footprint"));
    }
    Ok(())
}

/// Validates every kernel argument before a host service inspects the slice.
pub fn validate_kernel_arguments_v1(
    arguments: &[BundleKernelArgumentV1],
) -> Result<(), BundleAbiError> {
    for argument in arguments {
        validate_kernel_argument_v1(*argument)?;
    }
    Ok(())
}

/// Validates one custom operand before a host resolves the span it names.
///
/// Its declared length is exact rather than a range, for the reason
/// [`validate_custom_operand_struct_size_v1`] states: this record is an element
/// of the array `run_custom_operation` takes, and that callback carries no
/// element stride.
///
/// A flag bit this version does not define is refused rather than ignored: the
/// bit would ask the host for something it cannot do, and an operand whose use
/// the host misread is exactly the one an alias check would then miss.
pub fn validate_custom_operand_v1(operand: BundleCustomOperandV1) -> Result<(), BundleAbiError> {
    validate_custom_operand_struct_size_v1(operand.struct_size)?;
    if operand.flags & !BUNDLE_CUSTOM_OPERAND_FLAGS_V1_KNOWN != 0 {
        return Err(BundleAbiError::InvalidCustomOperandFlags(operand.flags));
    }
    validate_device_buffer_view_v1(operand.buffer)
}

/// Validates every custom operand before a host service inspects the slice.
pub fn validate_custom_operands_v1(
    operands: &[BundleCustomOperandV1],
) -> Result<(), BundleAbiError> {
    for operand in operands {
        validate_custom_operand_v1(*operand)?;
    }
    Ok(())
}

/// Validates a model-neutral physical KV layout and its byte-view identifier.
///
/// Its declared length is exact rather than a range. This record is embedded by
/// value in the physical KV request, whose `page_capacity` and
/// `request_capacity` sit after it, so a tail here would move both.
pub fn validate_physical_kv_layout_v1(
    layout: BundlePhysicalKvLayoutV1,
) -> Result<(), BundleAbiError> {
    validate_physical_kv_layout_struct_size_v1(layout.struct_size)?;
    if layout.rank_count == 0 || layout.page_bytes == 0 || layout.page_tokens == 0 {
        return Err(BundleAbiError::InvalidRange("physical KV layout"));
    }
    validate_alignment("physical KV layout", layout.page_alignment)?;
    validate_byte_view_v1(layout.layout_id)
}

/// Validates a physical KV request and checks its total requested byte range.
pub fn validate_physical_kv_request_v1(
    request: BundlePhysicalKvRequestV1,
) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "physical KV request",
        request.struct_size,
        struct_size_v1::<BundlePhysicalKvRequestV1>(),
    )?;
    validate_physical_kv_layout_v1(request.layout)?;
    request
        .page_capacity
        .checked_mul(request.layout.page_bytes)
        .ok_or(BundleAbiError::InvalidRange("physical KV request"))?;
    Ok(())
}

/// Validates a persistent-state request and its byte-view identifier.
pub fn validate_state_request_v1(request: BundleStateRequestV1) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "state request",
        request.struct_size,
        struct_size_v1::<BundleStateRequestV1>(),
    )?;
    validate_alignment("state request", request.alignment)?;
    validate_byte_view_v1(request.state_id)
}

/// Validates a collective request and its model-neutral semantic contract bytes.
pub fn validate_collective_request_v1(
    request: BundleCollectiveRequestV1,
) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "collective request",
        request.struct_size,
        struct_size_v1::<BundleCollectiveRequestV1>(),
    )?;
    if request.rank_count == 0 {
        return Err(BundleAbiError::InvalidRange("collective request"));
    }
    validate_byte_view_v1(request.semantic_contract)
}

pub fn validate_execute_logits_row_v1(row: BundleLogitsRowV1) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "logits row",
        row.struct_size,
        struct_size_v1::<BundleLogitsRowV1>(),
    )?;
    validate_reserved("logits row", row.reserved)?;
    if row.logit_count == 0 {
        return Err(BundleAbiError::InvalidLogitsRow);
    }
    let expected_length = row
        .logit_count
        .checked_mul(4)
        .ok_or(BundleAbiError::InvalidLogitsRow)?;
    if row.logit_bits.length != expected_length {
        return Err(BundleAbiError::InvalidLogitsRow);
    }
    validate_byte_view_v1(row.logit_bits)
}

pub fn validate_draft_tokens_v1(record: BundleDraftTokensV1) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "draft tokens",
        record.struct_size,
        struct_size_v1::<BundleDraftTokensV1>(),
    )?;
    validate_reserved("draft tokens", record.reserved)?;
    if record.token_count == 0
        || record.token_count > 1 << 20
        || record.token_bytes.length != u64::from(record.token_count) * 4
    {
        return Err(BundleAbiError::InvalidDraftTokens);
    }
    validate_byte_view_v1(record.token_bytes)
}

pub fn validate_kept_rows_v1(record: BundleKeptRowsV1) -> Result<(), BundleAbiError> {
    if record.struct_size != struct_size_v1::<BundleKeptRowsV1>() {
        return Err(BundleAbiError::StructSizeExact {
            surface: "kept rows",
            expected: struct_size_v1::<BundleKeptRowsV1>(),
            actual: record.struct_size,
        });
    }
    validate_reserved("kept rows", record.reserved)?;
    validate_reserved("kept rows tail", record.reserved_tail)?;
    if record.kept_rows == 0 {
        return Err(BundleAbiError::InvalidKeptRows);
    }
    Ok(())
}

pub fn validate_speculation_v1(record: BundleRawSpeculationV1) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "speculation",
        record.struct_size,
        struct_size_v1::<BundleRawSpeculationV1>(),
    )?;
    if record.maximum_draft_tokens == 0 {
        if record.maximum_verify_tokens != 0 {
            return Err(BundleAbiError::InvalidRange("speculation"));
        }
    } else if record.maximum_verify_tokens < record.maximum_draft_tokens.saturating_add(1) {
        return Err(BundleAbiError::InvalidRange("speculation"));
    }
    Ok(())
}

pub fn validate_batch_limits_v1(record: BundleRawBatchLimitsV1) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "batch limits",
        record.struct_size,
        struct_size_v1::<BundleRawBatchLimitsV1>(),
    )
}

pub fn validate_cluster_residency_v1(
    record: BundleRawClusterResidencyV1,
) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "cluster residency",
        record.struct_size,
        struct_size_v1::<BundleRawClusterResidencyV1>(),
    )?;
    if record.flags
        & !(BUNDLE_CLUSTER_RESIDENCY_V1_MPS_CLIENT
            | BUNDLE_CLUSTER_RESIDENCY_V1_CLUSTERS_REPORTED
            | BUNDLE_CLUSTER_RESIDENCY_V1_COOPERATIVE_LAUNCH)
        != 0
    {
        return Err(BundleAbiError::InvalidRange("cluster residency flags"));
    }
    Ok(())
}

/// Validates one ordered device view before the host resolves its buffer.
pub fn validate_execute_logits_view_v1(view: BundleLogitsViewV1) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "device logits view",
        view.struct_size,
        struct_size_v1::<BundleLogitsViewV1>(),
    )?;
    validate_reserved("device logits view", view.reserved)?;
    validate_reserved("device logits view tail", view.reserved_tail)?;
    view.context.validate()?;
    validate_device_buffer_view_v1(view.buffer)?;
    let bytes_per_element = match view.element_type {
        BUNDLE_LOGITS_ELEMENT_V1_F32 => 4_u64,
        BUNDLE_LOGITS_ELEMENT_V1_BF16 => 2_u64,
        _ => return Err(BundleAbiError::InvalidLogitsView),
    };
    if view.row_count == 0
        || view.row_count > 64
        || view.live_width == 0
        || view.live_width > 1 << 20
        || view.row_stride_elements < u64::from(view.live_width)
    {
        return Err(BundleAbiError::InvalidLogitsView);
    }
    let needed_elements = u64::from(view.row_count - 1)
        .checked_mul(view.row_stride_elements)
        .and_then(|prefix| prefix.checked_add(u64::from(view.live_width)))
        .ok_or(BundleAbiError::InvalidLogitsView)?;
    let needed_bytes = needed_elements
        .checked_mul(bytes_per_element)
        .ok_or(BundleAbiError::InvalidLogitsView)?;
    if needed_bytes > view.buffer.length
        || view.first_item_index.checked_add(view.row_count).is_none()
    {
        return Err(BundleAbiError::InvalidLogitsView);
    }
    Ok(())
}

pub fn validate_execute_result_sink_v1(
    sink: BundleExecuteResultSinkV1,
) -> Result<(), BundleAbiError> {
    validate_struct_size(
        "execute result sink",
        sink.struct_size,
        std::mem::offset_of!(BundleExecuteResultSinkV1, emit_logits_view) as u32,
    )?;
    validate_reserved("execute result sink", sink.reserved)?;
    if sink.emit_logits_row.is_none() {
        return Err(BundleAbiError::MissingExecuteResultCallback);
    }
    Ok(())
}

fn validate_struct_size(
    surface: &'static str,
    actual: u32,
    expected: u32,
) -> Result<(), BundleAbiError> {
    if actual < expected {
        Err(BundleAbiError::StructSize {
            surface,
            expected,
            actual,
        })
    } else {
        Ok(())
    }
}

fn validate_reserved(surface: &'static str, actual: u32) -> Result<(), BundleAbiError> {
    if actual == 0 {
        Ok(())
    } else {
        Err(BundleAbiError::InvalidReserved { surface, actual })
    }
}

fn validate_alignment(surface: &'static str, actual: u64) -> Result<(), BundleAbiError> {
    if actual != 0 && actual.is_power_of_two() {
        Ok(())
    } else {
        Err(BundleAbiError::InvalidAlignment { surface, actual })
    }
}

fn validate_range(surface: &'static str, offset: u64, length: u64) -> Result<(), BundleAbiError> {
    offset
        .checked_add(length)
        .ok_or(BundleAbiError::InvalidRange(surface))?;
    Ok(())
}

fn validate_table_header(
    surface: &'static str,
    struct_size: u32,
    expected_size: u32,
    abi_version: u32,
) -> Result<(), BundleAbiError> {
    validate_struct_size(surface, struct_size, expected_size)?;
    if abi_version != BUNDLE_ABI_VERSION_V1 {
        return Err(BundleAbiError::AbiVersion {
            surface,
            actual: abi_version,
        });
    }
    Ok(())
}

pub(crate) fn validate_bundle_api_header_v1(
    header: BundleApiHeaderV1,
) -> Result<(), BundleAbiError> {
    validate_table_header(
        "bundle API",
        header.struct_size,
        struct_size_v1::<BundleRawApiV1>(),
        header.abi_version,
    )
}

/// Checks one raw V1 service table without capturing or calling it.
pub fn validate_host_services_v1(raw: &BundleHostServicesV1) -> Result<(), BundleAbiError> {
    validate_table_header(
        "host services",
        raw.struct_size,
        std::mem::offset_of!(BundleHostServicesV1, acquire_serving_kv) as u32,
        raw.abi_version,
    )?;
    validate_host_services(raw)
}

fn validate_host_services(raw: &BundleHostServicesV1) -> Result<(), BundleAbiError> {
    macro_rules! require_service {
        ($field:ident) => {
            if raw.$field.is_none() {
                return Err(BundleAbiError::MissingHostService(stringify!($field)));
            }
        };
    }

    require_service!(log);
    require_service!(select_context);
    require_service!(release_context);
    require_service!(select_memory_pool);
    require_service!(set_memory_pool_peer_access);
    require_service!(release_memory_pool);
    require_service!(allocate_device_buffer);
    require_service!(release_device_buffer);
    require_service!(open_compiled_image);
    require_service!(upload_rank_arena);
    require_service!(release_rank_arena);
    require_service!(resolve_compiled_image_view);
    require_service!(release_compiled_image);
    require_service!(open_authenticated_component);
    require_service!(inspect_authenticated_component);
    require_service!(read_authenticated_component);
    require_service!(release_authenticated_component);
    require_service!(acquire_peer_access);
    require_service!(release_peer_access);
    require_service!(create_stream);
    require_service!(select_stage_stream);
    require_service!(synchronize_stream);
    require_service!(release_stream);
    require_service!(create_event);
    require_service!(record_event);
    require_service!(wait_event);
    require_service!(synchronize_event);
    require_service!(release_event);
    require_service!(copy_host_to_device);
    require_service!(copy_device_to_host);
    require_service!(copy_device_to_device);
    require_service!(copy_peer_to_peer);
    require_service!(zero_device_buffer);
    require_service!(write_device_pointer_table);
    require_service!(load_cuda_module);
    require_service!(resolve_cuda_symbol);
    require_service!(configure_cuda_symbol);
    require_service!(launch_cuda_symbol);
    require_service!(release_cuda_symbol);
    require_service!(release_cuda_module);
    require_service!(allocate_physical_kv);
    require_service!(create_kv_request);
    require_service!(prepare_kv);
    require_service!(commit_kv);
    require_service!(cancel_kv);
    require_service!(resolve_kv_view);
    require_service!(release_kv_request);
    require_service!(release_physical_kv);
    require_service!(allocate_state);
    require_service!(resolve_state_view);
    require_service!(release_state);
    require_service!(create_collective);
    require_service!(execute_collective);
    require_service!(release_collective);
    require_service!(run_custom_operation);
    require_service!(begin_graph_capture);
    require_service!(end_graph_capture);
    require_service!(launch_graph);
    require_service!(release_graph);
    Ok(())
}

/// A checked bundle API table with no ownership of the loaded native module.
#[derive(Clone, Copy)]
pub struct BundleApiV1 {
    raw: BundleRawApiV1,
    retire_request_v1: Option<BundleRetireRequestFnV1>,
    commit_accepted_v1: Option<BundleCommitAcceptedFnV1>,
    prefix_cache_v1: Option<PrefixCacheCallbacksV1>,
    restore_prefix_state_in_slot_v1: Option<BundleRestorePrefixStateInSlotFnV1>,
}

impl BundleApiV1 {
    /// Captures only the copied prefix, even when its size advertises a tail.
    pub fn new(raw: BundleRawApiV1) -> Result<Self, BundleAbiError> {
        validate_bundle_api_header_v1(raw.header())?;
        for (name, callback) in [
            ("create", raw.create.is_some()),
            ("execute", raw.execute.is_some()),
            ("commit", raw.commit.is_some()),
            ("cancel", raw.cancel.is_some()),
            ("destroy", raw.destroy.is_some()),
        ] {
            if !callback {
                return Err(BundleAbiError::MissingApiCallback(name));
            }
        }
        Ok(Self {
            raw,
            retire_request_v1: None,
            commit_accepted_v1: None,
            prefix_cache_v1: None,
            restore_prefix_state_in_slot_v1: None,
        })
    }

    /// Captures an explicitly supplied extension when its full field is advertised.
    pub fn new_with_request_retirement(
        raw: BundleApiWithRequestRetirementV1,
    ) -> Result<Self, BundleAbiError> {
        let mut api = Self::new(raw.base)?;
        if raw.base.struct_size as usize >= REQUEST_RETIREMENT_END_V1 {
            api.retire_request_v1 = raw.retire_request_v1;
        }
        Ok(api)
    }

    pub fn new_with_speculation(raw: BundleApiWithSpeculationV1) -> Result<Self, BundleAbiError> {
        let mut api = Self::new_with_request_retirement(raw.base)?;
        if raw.base.base.struct_size as usize >= COMMIT_ACCEPTED_END_V1 {
            api.commit_accepted_v1 = raw.commit_accepted_v1;
        }
        Ok(api)
    }

    pub fn new_with_prefix_cache(raw: BundleApiWithPrefixCacheV1) -> Result<Self, BundleAbiError> {
        let mut api = Self::new_with_speculation(raw.base)?;
        if raw.base.base.base.struct_size as usize >= PREFIX_CACHE_END_V1 {
            api.prefix_cache_v1 = PrefixCacheCallbacksV1::from_table(
                raw.describe_prefix_state_v1,
                raw.capture_prefix_state_v1,
                raw.restore_prefix_state_v1,
            );
        }
        Ok(api)
    }

    /// The state-slot restore counts only beside all three prefix callbacks.
    pub fn new_with_prefix_state_slot(
        raw: BundleApiWithPrefixStateSlotV1,
    ) -> Result<Self, BundleAbiError> {
        let mut api = Self::new_with_prefix_cache(raw.base)?;
        if raw.base.base.base.base.struct_size as usize >= PREFIX_STATE_SLOT_END_V1
            && api.prefix_cache_v1.is_some()
        {
            api.restore_prefix_state_in_slot_v1 = raw.restore_prefix_state_in_slot_v1;
        }
        Ok(api)
    }

    #[must_use]
    pub const fn supports_request_retirement(&self) -> bool {
        self.retire_request_v1.is_some()
    }

    #[must_use]
    pub const fn supports_speculation(&self) -> bool {
        self.commit_accepted_v1.is_some()
    }

    #[must_use]
    pub const fn supports_prefix_cache(&self) -> bool {
        self.prefix_cache_v1.is_some()
    }

    #[must_use]
    pub const fn supports_prefix_state_slot(&self) -> bool {
        self.restore_prefix_state_in_slot_v1.is_some()
    }

    #[must_use]
    pub const fn raw(&self) -> BundleRawApiV1 {
        self.raw
    }

    /// Creates one instance against a safe host-service table.
    ///
    /// The checks before the call are the only place this wrapper may release
    /// the table. Once the bundle's create callback runs it owns a caller, and
    /// nothing this wrapper can observe says whether it kept one.
    pub fn create(
        &self,
        host_services: HostServicesTableV1,
        configuration: &[u8],
    ) -> Result<BundleInstanceV1, BundleAbiError> {
        validate_host_services_v1(host_services.services())?;
        let create = self
            .raw
            .create
            .ok_or(BundleAbiError::MissingApiCallback("create"))?;
        let result = trampoline::invoke_bundle_create_v1(create, &host_services, configuration);
        match checked_create_result_v1(result) {
            Ok(handle) => Ok(BundleInstanceV1 {
                api: *self,
                handle,
                state: BundleInstanceStateV1::Ready,
                services: Some(host_services),
                last_execute_refusal: None,
            }),
            Err(error) => {
                retain_host_services_for_process_lifetime_v1(host_services);
                Err(error)
            }
        }
    }

    /// Runs one execution and delivers its rows to a safe host sink.
    ///
    /// A failed status leaves the caller with nothing to finalize, so a V1
    /// bundle owes it that there is nothing: see [`BundleInstanceV1::execute`]
    /// for what a bundle must do when it cannot close what it opened.
    pub fn execute_handle<S: LogitsRowSinkV1>(
        &self,
        handle: BundleHandleV1,
        request: &[u8],
        result_sink: &mut S,
    ) -> Result<(), BundleAbiError> {
        if !handle.has_generation() {
            return Err(BundleAbiError::InvalidHandle);
        }
        let execute = self
            .raw
            .execute
            .ok_or(BundleAbiError::MissingApiCallback("execute"))?;
        status_result(trampoline::invoke_bundle_execute_v1(
            execute,
            handle,
            request,
            result_sink,
        )?)
    }

    fn commit_handle(&self, handle: BundleHandleV1) -> Result<(), BundleAbiError> {
        let commit = self
            .raw
            .commit
            .ok_or(BundleAbiError::MissingApiCallback("commit"))?;
        status_result(commit(handle))
    }

    fn commit_accepted_handle(
        &self,
        handle: BundleHandleV1,
        records: &[BundleKeptRowsV1],
    ) -> Result<(), BundleAbiError> {
        if records.len() > 1 << 20 {
            return Err(BundleAbiError::InvalidKeptRows);
        }
        for record in records {
            validate_kept_rows_v1(*record)?;
        }
        let callback = self
            .commit_accepted_v1
            .ok_or(BundleAbiError::MissingApiCallback("commit_accepted_v1"))?;
        trampoline::invoke_bundle_commit_accepted_v1(callback, handle, records)
    }

    fn cancel_handle(&self, handle: BundleHandleV1) -> Result<(), BundleAbiError> {
        let cancel = self
            .raw
            .cancel
            .ok_or(BundleAbiError::MissingApiCallback("cancel"))?;
        status_result(cancel(handle))
    }

    fn destroy_handle(&self, handle: BundleHandleV1) -> Result<(), BundleAbiError> {
        let destroy = self
            .raw
            .destroy
            .ok_or(BundleAbiError::MissingApiCallback("destroy"))?;
        status_result(destroy(handle))
    }
}

fn status_result(status: BundleStatusV1) -> Result<(), BundleAbiError> {
    if status.is_success() {
        Ok(())
    } else {
        Err(BundleAbiError::Status(status))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BundleInstanceStateV1 {
    Ready,
    Pending,
    Destroyed,
    Poisoned,
}

/// A generation-checked bundle instance with one active transaction at most.
///
/// A failed retirement, commit, cancel, or destroy leaves the native instance state unknown.
/// The wrapper marks it poisoned and does not call another lifecycle callback.
/// That is terminal here: no ABI client retries the callback that failed, and a
/// bundle that keeps those resources for the process does so for its own
/// safety, not to offer this wrapper a second attempt.
///
/// Dropping the instance, after a successful destroy or after poisoning, hands
/// its service table to process-lifetime retention. The bundle may still hold
/// the caller that table produced, so the callbacks and the `user_data`
/// allocation behind it are never freed.
pub struct BundleInstanceV1 {
    api: BundleApiV1,
    handle: BundleHandleV1,
    state: BundleInstanceStateV1,
    services: Option<HostServicesTableV1>,
    last_execute_refusal: Option<String>,
}

struct ObservedExecuteSinkV1<'a, S> {
    sink: &'a mut S,
    emitted: bool,
}

impl<S: LogitsRowSinkV1> LogitsRowSinkV1 for ObservedExecuteSinkV1<'_, S> {
    fn limits(&self) -> ResultSinkLimitsV1 {
        self.sink.limits()
    }
    fn emit_row(&mut self, row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        self.emitted = true;
        self.sink.emit_row(row)
    }
    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        self.emitted = true;
        self.sink.emit_view(view)
    }
    fn emit_draft_tokens(&mut self, drafts: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        self.emitted = true;
        self.sink.emit_draft_tokens(drafts)
    }
}

impl BundleInstanceV1 {
    /// Diagnostic for a failed execute proven to have used only host logging.
    /// The V1 failed-execute contract guarantees no pending transaction.
    #[must_use]
    pub fn last_execute_refusal(&self) -> Option<&str> {
        self.last_execute_refusal.as_deref()
    }

    #[must_use]
    pub const fn handle(&self) -> BundleHandleV1 {
        self.handle
    }

    #[must_use]
    pub fn host_services(&self) -> &HostServicesTableV1 {
        self.services
            .as_ref()
            .expect("live bundle instance retains host services")
    }

    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.state == BundleInstanceStateV1::Poisoned
    }

    /// Runs one execution, leaving the instance pending only if it succeeded.
    ///
    /// The wrapper stays ready after a refusal and issues no finalizer, so a
    /// V1 bundle that reports failure must leave no transaction open: one it
    /// keeps is one no later commit or cancel can name. A bundle that cannot
    /// close what it opened owes the opposite report instead, a successful
    /// execution whose result is empty or otherwise unusable. This wrapper then
    /// enters its pending state, and the caller, finding nothing it can use,
    /// cancels through the same handle and reaches that transaction.
    pub fn execute<S: LogitsRowSinkV1>(
        &mut self,
        request: &[u8],
        result_sink: &mut S,
    ) -> Result<(), BundleAbiError> {
        self.last_execute_refusal = None;
        if self.is_poisoned() {
            return Err(BundleAbiError::Poisoned);
        }
        if self.state != BundleInstanceStateV1::Ready {
            return Err(BundleAbiError::Lifecycle("execute"));
        }
        self.host_services().begin_execute();
        let mut sink = ObservedExecuteSinkV1 {
            sink: result_sink,
            emitted: false,
        };
        let result = self.api.execute_handle(self.handle, request, &mut sink);
        let (service_called, diagnostic) = self.host_services().end_execute();
        if result == Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
            && !service_called
            && !sink.emitted
        {
            self.last_execute_refusal = Some(
                diagnostic
                    .filter(|text| !text.is_empty())
                    .unwrap_or_else(|| {
                        "execute argument validation (bundle supplied no check diagnostic)"
                            .to_owned()
                    }),
            );
        }
        result?;
        self.state = BundleInstanceStateV1::Pending;
        Ok(())
    }

    pub fn commit(&mut self) -> Result<(), BundleAbiError> {
        if self.is_poisoned() {
            return Err(BundleAbiError::Poisoned);
        }
        if self.state != BundleInstanceStateV1::Pending {
            return Err(BundleAbiError::Lifecycle("commit"));
        }
        match self.api.commit_handle(self.handle) {
            Ok(()) => {
                self.state = BundleInstanceStateV1::Ready;
                Ok(())
            }
            Err(error) => {
                self.state = BundleInstanceStateV1::Poisoned;
                Err(error)
            }
        }
    }

    /// Commits the accepted prefix of each verify item in item order.
    pub fn commit_accepted(&mut self, records: &[BundleKeptRowsV1]) -> Result<(), BundleAbiError> {
        if self.is_poisoned() {
            return Err(BundleAbiError::Poisoned);
        }
        if self.state != BundleInstanceStateV1::Pending {
            return Err(BundleAbiError::Lifecycle("commit_accepted"));
        }
        match self.api.commit_accepted_handle(self.handle, records) {
            Ok(()) => {
                self.state = BundleInstanceStateV1::Ready;
                Ok(())
            }
            Err(error) => {
                self.state = BundleInstanceStateV1::Poisoned;
                Err(error)
            }
        }
    }

    pub fn cancel(&mut self) -> Result<(), BundleAbiError> {
        if self.is_poisoned() {
            return Err(BundleAbiError::Poisoned);
        }
        if self.state != BundleInstanceStateV1::Pending {
            return Err(BundleAbiError::Lifecycle("cancel"));
        }
        match self.api.cancel_handle(self.handle) {
            Ok(()) => {
                self.state = BundleInstanceStateV1::Ready;
                Ok(())
            }
            Err(error) => {
                self.state = BundleInstanceStateV1::Poisoned;
                Err(error)
            }
        }
    }

    #[must_use]
    pub const fn supports_request_retirement(&self) -> bool {
        self.api.supports_request_retirement()
    }

    #[must_use]
    pub const fn supports_speculation(&self) -> bool {
        self.api.supports_speculation()
    }

    #[must_use]
    pub const fn supports_prefix_cache(&self) -> bool {
        self.api.supports_prefix_cache()
    }

    #[must_use]
    pub const fn supports_prefix_state_slot(&self) -> bool {
        self.api.supports_prefix_state_slot()
    }

    /// Asks how many serving-pool pages one snapshot needs.
    ///
    /// Like the other prefix-state calls it runs only while Ready, and a
    /// refusal does not poison the instance: the bundle promised to change
    /// nothing when it refuses. The exception is a capture or restore that
    /// reports INTERNAL_ERROR or panics: its device work may still access
    /// the pages, so the instance is poisoned and the caller must not reuse
    /// them.
    pub fn describe_prefix_state(&mut self) -> Result<u64, BundleAbiError> {
        let callbacks = self.prefix_state_callbacks("describe_prefix_state")?;
        let info =
            trampoline::invoke_bundle_describe_prefix_state_v1(callbacks.describe, self.handle);
        validate_struct_size(
            "prefix state info",
            info.struct_size,
            struct_size_v1::<BundlePrefixStateInfoV1>(),
        )?;
        status_result(info.status)?;
        Ok(info.state_pages)
    }

    /// Asks the bundle to snapshot a request's committed state into `pages`.
    pub fn capture_prefix_state(
        &mut self,
        request_id: u64,
        tokens: u64,
        pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        let callbacks = self.prefix_state_callbacks("capture_prefix_state")?;
        let result = trampoline::invoke_bundle_prefix_state_v1(
            callbacks.capture,
            self.handle,
            request_id,
            tokens,
            &[],
            pages,
        );
        self.poison_on_unconfirmed_prefix_state(result)
    }

    /// Asks the bundle to create a request's state from a snapshot in `pages`.
    pub fn restore_prefix_state(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
        pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        let callbacks = self.prefix_state_callbacks("restore_prefix_state")?;
        let result = trampoline::invoke_bundle_prefix_state_v1(
            callbacks.restore,
            self.handle,
            request_id,
            token_ids.len() as u64,
            token_ids,
            pages,
        );
        self.poison_on_unconfirmed_prefix_state(result)
    }

    fn poison_on_unconfirmed_prefix_state(
        &mut self,
        result: Result<(), BundleAbiError>,
    ) -> Result<(), BundleAbiError> {
        if matches!(
            result,
            Err(BundleAbiError::Status(
                BundleStatusV1::InternalError | BundleStatusV1::Panic
            ))
        ) {
            self.state = BundleInstanceStateV1::Poisoned;
        }
        result
    }

    /// Like `restore_prefix_state`, into the state slot the host assigned.
    pub fn restore_prefix_state_in_slot(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
        pages: &[u64],
        state_slot: u64,
    ) -> Result<(), BundleAbiError> {
        self.prefix_state_callbacks("restore_prefix_state_in_slot")?;
        let restore =
            self.api
                .restore_prefix_state_in_slot_v1
                .ok_or(BundleAbiError::MissingApiCallback(
                    "restore_prefix_state_in_slot",
                ))?;
        let result = trampoline::invoke_bundle_restore_prefix_state_in_slot_v1(
            restore,
            self.handle,
            token_ids,
            pages,
            request_id,
            state_slot,
        );
        self.poison_on_unconfirmed_prefix_state(result)
    }

    fn prefix_state_callbacks(
        &self,
        name: &'static str,
    ) -> Result<PrefixCacheCallbacksV1, BundleAbiError> {
        if self.is_poisoned() {
            return Err(BundleAbiError::Poisoned);
        }
        if self.state != BundleInstanceStateV1::Ready {
            return Err(BundleAbiError::Lifecycle(name));
        }
        self.api
            .prefix_cache_v1
            .ok_or(BundleAbiError::MissingApiCallback(name))
    }

    /// Retires request state synchronously while Ready. Legacy tables do nothing.
    /// A callback error poisons the instance and prevents any further callback.
    pub fn retire_request(&mut self, request_id: u64) -> Result<(), BundleAbiError> {
        if self.is_poisoned() {
            return Err(BundleAbiError::Poisoned);
        }
        if self.state != BundleInstanceStateV1::Ready {
            return Err(BundleAbiError::Lifecycle("retire_request"));
        }
        if let Some(retire) = self.api.retire_request_v1
            && let Err(error) = status_result(retire(self.handle, request_id))
        {
            self.state = BundleInstanceStateV1::Poisoned;
            return Err(error);
        }
        Ok(())
    }

    pub fn destroy(&mut self) -> Result<(), BundleAbiError> {
        if self.is_poisoned() {
            return Err(BundleAbiError::Poisoned);
        }
        if self.state != BundleInstanceStateV1::Ready {
            return Err(BundleAbiError::Lifecycle("destroy"));
        }
        match self.api.destroy_handle(self.handle) {
            Ok(()) => {
                self.state = BundleInstanceStateV1::Destroyed;
                Ok(())
            }
            Err(error) => {
                self.state = BundleInstanceStateV1::Poisoned;
                Err(error)
            }
        }
    }

    pub(crate) fn finalize_for_drop(&mut self) {
        if self.state == BundleInstanceStateV1::Pending {
            let _ = self.cancel();
        }
        if self.state == BundleInstanceStateV1::Ready {
            let _ = self.destroy();
        }
        // A destroyed instance releases its resources, not its service table. The
        // bundle may still hold the caller, so the table outlives the instance.
        if let Some(services) = self.services.take() {
            retain_host_services_for_process_lifetime_v1(services);
        }
    }
}

fn checked_create_result_v1(
    result: BundleCreateResultV1,
) -> Result<BundleHandleV1, BundleAbiError> {
    validate_struct_size(
        "create result",
        result.struct_size,
        struct_size_v1::<BundleCreateResultV1>(),
    )?;
    status_result(result.status)?;
    if !result.handle.has_generation() {
        return Err(BundleAbiError::InvalidHandle);
    }
    Ok(result.handle)
}

/// Pins one service table, its callbacks, and its backend for the process.
///
/// A bundle receives an owned caller from its create trampoline and may keep it
/// for as long as it runs, whatever status it then returns. This crate cannot
/// see those copies, so freeing the callback code or the `user_data` allocation
/// would leave them dangling. Every table a create callback has seen therefore
/// stays resident. Device and KV resources are unaffected: they are still
/// released in reverse order, and a released handle is still rejected by its
/// generation.
fn retain_host_services_for_process_lifetime_v1(services: HostServicesTableV1) {
    std::mem::forget(services);
}

impl Drop for BundleInstanceV1 {
    fn drop(&mut self) {
        self.finalize_for_drop();
    }
}

/// Cancellation drains work but cannot restore recurrent state. The host
/// retires every item of a step it cancels before any slot can be reused;
/// a request whose sampling fails instead commits with its step (one kept
/// row) and then retires alone.
pub const RECURRENT_STATE_IN_PLACE_CAPABILITY_V1: &str = "chop.recurrent-state.in-place.v1";

/// Instances serving at most 32 items bound verification to 32 rows and
/// the manifest maximum. Wider instances use the manifest maximum. This
/// preserves small-instance row budgets without model-specific host policy.
pub const SPECULATIVE_NARROW_ROWS_CAPABILITY_V1: &str = "chop.speculative-narrow-rows.v1";
