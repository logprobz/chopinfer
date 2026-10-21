//! The safe host-service surface implemented outside this crate.
//!
//! A host implements [`HostServicesBackendV1`] with owned Rust values and
//! closure-scoped borrows. The ABI trampolines decode every native pointer
//! before they call a backend method and encode every result afterwards, so a
//! host implementation contains no unsafe code and never receives an ABI
//! pointer.

use std::ffi::c_void;

use crate::trampoline::CustomOperationV1;
use crate::{
    BUNDLE_CUSTOM_ADDRESS_SPACE_V1_DEVICE, BUNDLE_CUSTOM_ADDRESS_SPACE_V1_HOST,
    BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE, BUNDLE_CUSTOM_OPERATION_FLAG_V1_MAY_ENQUEUE,
    BUNDLE_CUSTOM_OPERATION_FLAGS_V1_KNOWN, BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN8B,
    BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN16B, BUNDLE_TENSOR_MAP_ELEMENT_V1_16U6_ALIGN16B,
    BUNDLE_TENSOR_MAP_ELEMENT_V1_BF16, BUNDLE_TENSOR_MAP_ELEMENT_V1_F16,
    BUNDLE_TENSOR_MAP_ELEMENT_V1_F32, BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E4M3,
    BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E5M2, BUNDLE_TENSOR_MAP_ELEMENT_V1_I32,
    BUNDLE_TENSOR_MAP_ELEMENT_V1_U8, BUNDLE_TENSOR_MAP_ELEMENT_V1_U16,
    BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_32, BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_64,
    BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_128, BUNDLE_TENSOR_MAP_SWIZZLE_V1_NONE, BundleAbiError,
    BundleAuthenticatedComponentHandleV1, BundleAuthenticatedComponentMetadataV1,
    BundleCollectiveHandleV1, BundleCompiledImageHandleV1, BundleContextHandleV1,
    BundleCudaModuleHandleV1, BundleCudaSymbolHandleV1, BundleDeviceBufferHandleV1,
    BundleDeviceBufferViewV1, BundleEventHandleV1, BundleGraphHandleV1, BundleKvRequestHandleV1,
    BundleKvTransactionHandleV1, BundleLaunchConfigV1, BundleMemoryPoolHandleV1,
    BundlePeerAccessHandleV1, BundlePhysicalKvHandleV1, BundleRankArenaHandleV1,
    BundleRawCompiledImageViewV1, BundleStateHandleV1, BundleStatusV1, BundleStreamHandleV1,
    BundleTensorMapV1, struct_size_v1,
};

/// The result every safe host-service implementation returns.
pub type HostServiceResultV1<T> = Result<T, BundleStatusV1>;

/// Metadata copied from one package-authenticated component handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AuthenticatedComponentMetadataV1 {
    pub role_length: u64,
    pub byte_length: u64,
    pub sha256: [u8; 64],
}

impl AuthenticatedComponentMetadataV1 {
    #[must_use]
    pub const fn new(role_length: u64, byte_length: u64, sha256: [u8; 64]) -> Self {
        Self {
            role_length,
            byte_length,
            sha256,
        }
    }

    pub(crate) const fn to_raw(self) -> BundleAuthenticatedComponentMetadataV1 {
        BundleAuthenticatedComponentMetadataV1 {
            struct_size: struct_size_v1::<BundleAuthenticatedComponentMetadataV1>(),
            reserved: 0,
            role_length: self.role_length,
            byte_length: self.byte_length,
            sha256: self.sha256,
        }
    }
}

/// One gather of rows of a host-mapped view
/// ([`HostServicesBackendV1::gather_mapped_rows`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MappedRowGatherV1<'a> {
    pub image: BundleCompiledImageHandleV1,
    pub rank: u32,
    pub view_id: &'a [u8],
    pub row_bytes: u64,
    pub rows: &'a [u32],
}

/// Immutable planes indexed by the same row ID, eligible for a shared host cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MappedRowPlaneV1 {
    pub image: BundleCompiledImageHandleV1,
    pub rank: u32,
    pub view_id: Vec<u8>,
    pub row_bytes: u64,
}

/// One plane of a prepared gather. Device U32 indices begin at `index_offset`
/// bytes of the shared input span. Every enqueue gathers the same row count
/// from each plane into the beginning of its bounded destination.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MappedGatherPlaneV1 {
    pub image: BundleCompiledImageHandleV1,
    pub rank: u32,
    pub view_id: Vec<u8>,
    pub row_bytes: u64,
    pub index_offset: u64,
    pub stream: BundleStreamHandleV1,
    pub destination: DeviceBufferSpanV1,
}

/// Timing of a completed gather. `exposed_nanoseconds` is the maximum
/// consumer-stream wait, after work queued before the dependency.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MappedGatherStatsV1 {
    pub host_nanoseconds: u64,
    pub exposed_nanoseconds: u64,
}

/// A device-buffer range that carries no ABI pointer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeviceBufferSpanV1 {
    pub buffer: BundleDeviceBufferHandleV1,
    pub offset: u64,
    pub length: u64,
}

impl DeviceBufferSpanV1 {
    #[must_use]
    pub const fn new(buffer: BundleDeviceBufferHandleV1, offset: u64, length: u64) -> Self {
        Self {
            buffer,
            offset,
            length,
        }
    }

    pub(crate) const fn from_view(view: BundleDeviceBufferViewV1) -> Self {
        Self {
            buffer: view.buffer,
            offset: view.offset,
            length: view.length,
        }
    }

    pub(crate) const fn to_view(self) -> BundleDeviceBufferViewV1 {
        BundleDeviceBufferViewV1 {
            struct_size: struct_size_v1::<BundleDeviceBufferViewV1>(),
            reserved: 0,
            buffer: self.buffer,
            offset: self.offset,
            length: self.length,
        }
    }
}

/// One rank's compiled-image range that carries no ABI pointer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompiledImageSpanV1 {
    pub rank: u32,
    pub buffer: DeviceBufferSpanV1,
    pub alignment: u64,
}

/// One rank's inputs to a batched compiled-image upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RankArenaUploadV1 {
    pub context: BundleContextHandleV1,
    pub pool: BundleMemoryPoolHandleV1,
    pub stream: BundleStreamHandleV1,
    pub image: BundleCompiledImageHandleV1,
    pub rank: u32,
}

impl CompiledImageSpanV1 {
    #[must_use]
    pub const fn new(rank: u32, buffer: DeviceBufferSpanV1, alignment: u64) -> Self {
        Self {
            rank,
            buffer,
            alignment,
        }
    }

    pub(crate) const fn to_view(self) -> BundleRawCompiledImageViewV1 {
        BundleRawCompiledImageViewV1 {
            struct_size: struct_size_v1::<BundleRawCompiledImageViewV1>(),
            rank: self.rank,
            buffer: self.buffer.to_view(),
            alignment: self.alignment,
        }
    }
}

/// The thread-block cluster one launch groups its blocks into.
///
/// [`LaunchClusterV1::NONE`] is one block in every dimension, which asks for no
/// cluster and is the ordinary launch. Any other shape is exact: its blocks
/// multiply out to at most [`BUNDLE_LAUNCH_V1_MAX_CLUSTER_BLOCKS`], and
/// each grid extent divides by the cluster extent beside it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchClusterV1 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl LaunchClusterV1 {
    /// The one-block cluster, which is the launch that asks for no cluster.
    pub const NONE: Self = Self { x: 1, y: 1, z: 1 };

    /// Whether this shape asks a host for a cluster launch at all.
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.x == 1 && self.y == 1 && self.z == 1
    }

    /// The blocks one cluster of this shape holds, or `None` if it overflows.
    #[must_use]
    pub const fn blocks(self) -> Option<u32> {
        match self.x.checked_mul(self.y) {
            Some(blocks) => blocks.checked_mul(self.z),
            None => None,
        }
    }
}

/// A validated CUDA launch shape with no model-specific limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LaunchConfigV1 {
    pub flags: u32,
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
    pub shared_memory_bytes: u32,
    pub cluster: LaunchClusterV1,
}

impl LaunchConfigV1 {
    pub(crate) const fn to_config(self) -> BundleLaunchConfigV1 {
        BundleLaunchConfigV1 {
            struct_size: struct_size_v1::<BundleLaunchConfigV1>(),
            flags: self.flags,
            grid_x: self.grid_x,
            grid_y: self.grid_y,
            grid_z: self.grid_z,
            block_x: self.block_x,
            block_y: self.block_y,
            block_z: self.block_z,
            shared_memory_bytes: self.shared_memory_bytes,
            cluster_x: self.cluster.x,
            cluster_y: self.cluster.y,
            cluster_z: self.cluster.z,
        }
    }

    pub(crate) const fn from_config(config: BundleLaunchConfigV1) -> Self {
        Self {
            flags: config.flags,
            grid_x: config.grid_x,
            grid_y: config.grid_y,
            grid_z: config.grid_z,
            block_x: config.block_x,
            block_y: config.block_y,
            block_z: config.block_z,
            shared_memory_bytes: config.shared_memory_bytes,
            cluster: LaunchClusterV1 {
                x: config.cluster_x,
                y: config.cluster_y,
                z: config.cluster_z,
            },
        }
    }
}

/// The element type a tensor map is declared over.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorMapElementV1 {
    U8,
    U16,
    I32,
    Bf16,
    F16,
    F32,
    Fp8E4M3,
    Fp8E5M2,
    U4Align8B,
    U4Align16B,
    U6Align16B,
}

impl TensorMapElementV1 {
    /// The bytes one element of this type occupies.
    #[must_use]
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::U8 | Self::Fp8E4M3 | Self::Fp8E5M2 | Self::U4Align16B | Self::U6Align16B => 1,
            Self::U4Align8B => 0,
            Self::I32 | Self::F32 => 4,
            Self::Bf16 | Self::F16 | Self::U16 => 2,
        }
    }

    pub const fn code(self) -> u32 {
        match self {
            Self::U8 => BUNDLE_TENSOR_MAP_ELEMENT_V1_U8,
            Self::U16 => BUNDLE_TENSOR_MAP_ELEMENT_V1_U16,
            Self::I32 => BUNDLE_TENSOR_MAP_ELEMENT_V1_I32,
            Self::Bf16 => BUNDLE_TENSOR_MAP_ELEMENT_V1_BF16,
            Self::F16 => BUNDLE_TENSOR_MAP_ELEMENT_V1_F16,
            Self::F32 => BUNDLE_TENSOR_MAP_ELEMENT_V1_F32,
            Self::Fp8E4M3 => BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E4M3,
            Self::Fp8E5M2 => BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E5M2,
            Self::U4Align8B => BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN8B,
            Self::U4Align16B => BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN16B,
            Self::U6Align16B => BUNDLE_TENSOR_MAP_ELEMENT_V1_16U6_ALIGN16B,
        }
    }

    pub(crate) const fn from_code(code: u32) -> Result<Self, BundleAbiError> {
        match code {
            BUNDLE_TENSOR_MAP_ELEMENT_V1_U8 => Ok(Self::U8),
            BUNDLE_TENSOR_MAP_ELEMENT_V1_U16 => Ok(Self::U16),
            BUNDLE_TENSOR_MAP_ELEMENT_V1_I32 => Ok(Self::I32),
            BUNDLE_TENSOR_MAP_ELEMENT_V1_BF16 => Ok(Self::Bf16),
            BUNDLE_TENSOR_MAP_ELEMENT_V1_F16 => Ok(Self::F16),
            BUNDLE_TENSOR_MAP_ELEMENT_V1_F32 => Ok(Self::F32),
            BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E4M3 => Ok(Self::Fp8E4M3),
            BUNDLE_TENSOR_MAP_ELEMENT_V1_FP8_E5M2 => Ok(Self::Fp8E5M2),
            BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN8B => Ok(Self::U4Align8B),
            BUNDLE_TENSOR_MAP_ELEMENT_V1_16U4_ALIGN16B => Ok(Self::U4Align16B),
            BUNDLE_TENSOR_MAP_ELEMENT_V1_16U6_ALIGN16B => Ok(Self::U6Align16B),
            code => Err(BundleAbiError::InvalidTensorMapElement(code)),
        }
    }
}

/// The swizzle one tensor map tiles its innermost dimension with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorMapSwizzleV1 {
    None,
    Bytes32,
    Bytes64,
    Bytes128,
}

impl TensorMapSwizzleV1 {
    pub const fn code(self) -> u32 {
        match self {
            Self::None => BUNDLE_TENSOR_MAP_SWIZZLE_V1_NONE,
            Self::Bytes32 => BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_32,
            Self::Bytes64 => BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_64,
            Self::Bytes128 => BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_128,
        }
    }

    pub(crate) const fn from_code(code: u32) -> Result<Self, BundleAbiError> {
        match code {
            BUNDLE_TENSOR_MAP_SWIZZLE_V1_NONE => Ok(Self::None),
            BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_32 => Ok(Self::Bytes32),
            BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_64 => Ok(Self::Bytes64),
            BUNDLE_TENSOR_MAP_SWIZZLE_V1_BYTES_128 => Ok(Self::Bytes128),
            code => Err(BundleAbiError::InvalidTensorMapSwizzle(code)),
        }
    }
}

/// One tiled tensor map over a span the host already issued.
///
/// The map carries no address. `buffer` is the span it is encoded over, and
/// resolving that span to a device address is the host's work, not the
/// bundle's.
///
/// The C header defines the rank, geometry, layout, and packing limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorMapV1 {
    pub buffer: DeviceBufferSpanV1,
    pub rank: u32,
    pub element_type: TensorMapElementV1,
    pub swizzle: TensorMapSwizzleV1,
    pub interleave: u32,
    pub l2_promotion: u32,
    pub oob_fill: u32,
    pub global_dims: [u64; 5],
    pub global_strides_bytes: [u64; 4],
    pub box_dims: [u32; 5],
    pub element_strides: [u32; 5],
}

impl TensorMapV1 {
    pub(crate) fn to_raw(self) -> BundleTensorMapV1 {
        BundleTensorMapV1 {
            struct_size: struct_size_v1::<BundleTensorMapV1>(),
            rank: self.rank,
            element_type: self.element_type.code(),
            swizzle: self.swizzle.code(),
            interleave: self.interleave,
            l2_promotion: self.l2_promotion,
            oob_fill: self.oob_fill,
            reserved: 0,
            buffer: self.buffer.to_view(),
            global_dims: self.global_dims,
            global_strides_bytes: self.global_strides_bytes,
            box_dims: self.box_dims,
            element_strides: self.element_strides,
        }
    }

    pub(crate) fn from_raw(raw: BundleTensorMapV1) -> Result<Self, BundleAbiError> {
        Ok(Self {
            buffer: DeviceBufferSpanV1::from_view(raw.buffer),
            rank: raw.rank,
            element_type: TensorMapElementV1::from_code(raw.element_type)?,
            swizzle: TensorMapSwizzleV1::from_code(raw.swizzle)?,
            interleave: raw.interleave,
            l2_promotion: raw.l2_promotion,
            oob_fill: raw.oob_fill,
            global_dims: raw.global_dims,
            global_strides_bytes: raw.global_strides_bytes,
            box_dims: raw.box_dims,
            element_strides: raw.element_strides,
        })
    }
}

/// One kernel argument whose opaque bytes are copied out of ABI storage.
///
/// Each scalar launches at exactly the width it names. `OpaqueBytes` stays
/// opaque: V1 declares no launch encoding for it, so a host asked to launch one
/// reports it unsupported rather than choosing a width.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum KernelArgumentV1 {
    DeviceBuffer(DeviceBufferSpanV1),
    /// The null device pointer an unused slot of a fixed kernel ABI takes.
    NullDevicePointer,
    ScalarU32(u32),
    ScalarI32(i32),
    ScalarU64(u64),
    OpaqueBytes(Vec<u8>),
    TensorMap(TensorMapV1),
    /// A by-value array of addresses the host resolves, in slot order.
    ///
    /// `None` is the null pointer that a fixed kernel ABI's unused slot takes.
    /// The vector is what the argument carries, so its length is the slot count
    /// and the ABI refuses one that is empty or past the bound.
    DevicePointerArray(Vec<Option<DeviceBufferSpanV1>>),
}

/// One span a custom operation is lent, and whether it writes to it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CustomOperandV1 {
    pub span: DeviceBufferSpanV1,
    pub writable: bool,
}

impl CustomOperandV1 {
    #[must_use]
    pub const fn read(span: DeviceBufferSpanV1) -> Self {
        Self {
            span,
            writable: false,
        }
    }

    #[must_use]
    pub const fn write(span: DeviceBufferSpanV1) -> Self {
        Self {
            span,
            writable: true,
        }
    }

    pub(crate) const fn flags(self) -> u32 {
        if self.writable {
            BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE
        } else {
            0
        }
    }

    pub(crate) const fn from_flags(span: DeviceBufferSpanV1, flags: u32) -> Self {
        Self {
            span,
            writable: flags & BUNDLE_CUSTOM_OPERAND_FLAG_V1_WRITABLE != 0,
        }
    }
}

/// What one custom operation may leave behind on the stream it is lent.
///
/// The set is closed, and it decides one thing: whether the host waits after
/// the callback as well as before it. A synchronous operation finished
/// everything it started, so there is nothing left to wait for and nothing a
/// failed wait could split. An operation that may enqueue is waited for, and
/// until that wait succeeds nothing it was lent may be released.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CustomCompletionV1 {
    Synchronous,
    MayEnqueue,
}

impl CustomCompletionV1 {
    /// Whether the host waits for this operation after its callback returns.
    #[must_use]
    pub const fn waits_after_callback(self) -> bool {
        matches!(self, Self::MayEnqueue)
    }

    #[must_use]
    pub const fn flags(self) -> u32 {
        match self {
            Self::Synchronous => 0,
            Self::MayEnqueue => BUNDLE_CUSTOM_OPERATION_FLAG_V1_MAY_ENQUEUE,
        }
    }

    /// Reads one completion policy, refusing a bit this version cannot honour.
    pub const fn from_flags(flags: u32) -> Result<Self, BundleAbiError> {
        if flags & !BUNDLE_CUSTOM_OPERATION_FLAGS_V1_KNOWN != 0 {
            return Err(BundleAbiError::InvalidCustomCompletion(flags));
        }
        if flags & BUNDLE_CUSTOM_OPERATION_FLAG_V1_MAY_ENQUEUE != 0 {
            Ok(Self::MayEnqueue)
        } else {
            Ok(Self::Synchronous)
        }
    }
}

/// Which memory the raw values of one lent launch belong to.
///
/// A host with a driver lends device addresses and its own driver stream. A
/// host without one lends host addresses and a null stream. An operation that
/// needs one of them refuses the other rather than running against memory its
/// kernels cannot reach.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CustomAddressSpaceV1 {
    Device,
    Host,
}

impl CustomAddressSpaceV1 {
    pub(crate) const fn code(self) -> u32 {
        match self {
            Self::Device => BUNDLE_CUSTOM_ADDRESS_SPACE_V1_DEVICE,
            Self::Host => BUNDLE_CUSTOM_ADDRESS_SPACE_V1_HOST,
        }
    }

    pub(crate) const fn from_code(code: u32) -> Result<Self, BundleAbiError> {
        match code {
            BUNDLE_CUSTOM_ADDRESS_SPACE_V1_DEVICE => Ok(Self::Device),
            BUNDLE_CUSTOM_ADDRESS_SPACE_V1_HOST => Ok(Self::Host),
            code => Err(BundleAbiError::InvalidCustomAddressSpace(code)),
        }
    }
}

/// One range the host resolved to a raw address for exactly one call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LentRangeV1 {
    pub address: u64,
    pub length: u64,
}

impl LentRangeV1 {
    #[must_use]
    pub const fn new(address: u64, length: u64) -> Self {
        Self { address, length }
    }
}

/// The raw driver stream a host lends for exactly one custom-operation call.
///
/// A host without a driver lends [`LentStreamV1::null`], which is what its
/// address space already says.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LentStreamV1(*mut c_void);

impl LentStreamV1 {
    #[must_use]
    pub const fn from_raw(stream: *mut c_void) -> Self {
        Self(stream)
    }

    #[must_use]
    pub const fn null() -> Self {
        Self(std::ptr::null_mut())
    }

    pub(crate) const fn raw(self) -> *mut c_void {
        self.0
    }
}

/// Everything one host lends a custom operation for the length of one call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LentLaunchV1<'a> {
    pub address_space: CustomAddressSpaceV1,
    pub stream: LentStreamV1,
    pub ranges: &'a [LentRangeV1],
}

/// A model-neutral physical KV page layout with an owned layout identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalKvLayoutV1 {
    pub flags: u32,
    pub rank_count: u64,
    pub page_bytes: u64,
    pub page_alignment: u64,
    pub page_tokens: u64,
    pub state_lifetime: u64,
    pub layout_id: Vec<u8>,
}

/// A physical KV allocation request with an owned layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicalKvRequestV1 {
    pub flags: u32,
    pub layout: PhysicalKvLayoutV1,
    pub page_capacity: u64,
    pub request_capacity: u64,
}

/// A persistent-state request with an owned state identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateRequestV1 {
    pub flags: u32,
    pub byte_length: u64,
    pub alignment: u64,
    pub state_id: Vec<u8>,
}

/// A collective request with owned semantic-contract bytes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CollectiveRequestV1 {
    pub flags: u32,
    pub rank_count: u64,
    pub semantic_contract: Vec<u8>,
}

/// One logits row copied out of ABI storage before the emitting call returns.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnedLogitsRowV1 {
    request_id: u64,
    logit_bits: Vec<u8>,
}

/// Draft tokens copied from an ABI callback before its borrowed bytes expire.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OwnedDraftTokensV1 {
    pub request_id: u64,
    pub tokens: Vec<u32>,
}

/// A validated device view copied from a result callback. Its handle stays
/// opaque until the host resolves it against the live context and buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OwnedLogitsViewV1 {
    raw: crate::BundleLogitsViewV1,
}

impl OwnedLogitsViewV1 {
    #[must_use]
    pub const fn new(raw: crate::BundleLogitsViewV1) -> Self {
        Self { raw }
    }

    #[must_use]
    pub const fn raw(self) -> crate::BundleLogitsViewV1 {
        self.raw
    }
}

impl OwnedLogitsRowV1 {
    /// Builds one row from a request identifier and its little-endian logits.
    ///
    /// The ABI builds rows this way from the bytes a bundle emitted. It is
    /// public so a host or a conformance fake can build the same rows without a
    /// device, and it interprets neither field.
    #[must_use]
    pub const fn new(request_id: u64, logit_bits: Vec<u8>) -> Self {
        Self {
            request_id,
            logit_bits,
        }
    }

    #[must_use]
    pub const fn request_id(&self) -> u64 {
        self.request_id
    }

    #[must_use]
    pub fn logit_bits(&self) -> &[u8] {
        &self.logit_bits
    }

    #[must_use]
    pub fn logit_count(&self) -> u64 {
        u64::try_from(self.logit_bits.len() / 4).expect("owned logits row count fits u64")
    }
}

/// The bounded capacity one execute call may deliver through a result sink.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultSinkLimitsV1 {
    pub maximum_rows: u64,
    pub maximum_logits_per_row: u64,
    pub maximum_total_logits: u64,
}

impl ResultSinkLimitsV1 {
    #[must_use]
    pub const fn new(
        maximum_rows: u64,
        maximum_logits_per_row: u64,
        maximum_total_logits: u64,
    ) -> Self {
        Self {
            maximum_rows,
            maximum_logits_per_row,
            maximum_total_logits,
        }
    }
}

/// The safe destination for the logits rows one execute call emits.
///
/// The ABI copies each row into owned storage before this trait sees it, so a
/// host sink never holds an ABI pointer.
pub trait LogitsRowSinkV1 {
    /// Explicit opt-in to the separate device-result execute export.
    fn device_results_enabled(&self) -> bool {
        false
    }
    fn device_cycle_metadata(
        &self,
    ) -> HostServiceResultV1<crate::device_results::DeviceCycleMetadataV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn declare_input_dependency(&mut self, _dependency: u32) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn emit_device_drafts(
        &mut self,
        _drafts: crate::device_results::BundleDeviceDraftsV1,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn sample_device_logits(
        &mut self,
        _request: crate::device_results::BundleDeviceSamplingRequestV1,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn emit_ready_logits_view(
        &mut self,
        _view: crate::device_results::BundleReadyLogitsViewV1,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn emit_device_completion(
        &mut self,
        _completion: crate::device_results::BundleDeviceCompletionV1,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Reports the bounds the ABI enforces before it forwards any row.
    fn limits(&self) -> ResultSinkLimitsV1;

    /// Accepts one copied row that arrived before the transaction was finalized.
    fn emit_row(&mut self, row: OwnedLogitsRowV1) -> HostServiceResultV1<()>;

    /// Accepts a validated device view while the referenced buffer is live.
    fn emit_view(&mut self, _view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn emit_draft_tokens(&mut self, _drafts: OwnedDraftTokensV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
}

/// Every model-neutral service a host provides to one bundle instance.
///
/// Implementations receive owned records, copied byte vectors, or borrows that
/// end when the method returns. They never receive an ABI pointer and never
/// need unsafe code.
pub trait HostServicesBackendV1 {
    fn log(&self, message: &[u8]) -> HostServiceResultV1<()>;

    /// Selects the one context a rank has.
    ///
    /// A rank is one driver context, so selecting it twice returns the same
    /// handle rather than a second name for one object, and each selection is
    /// one user: the context goes back to the driver when the last selection
    /// releases it.
    ///
    /// That makes a context handle an exact name for the driver context
    /// underneath, which is what a peer lease is keyed by, and it holds because
    /// the host validates its topology first: two ranks may not name one device
    /// ordinal, so no two ranks share a primary context.
    fn select_context(
        &self,
        rank: u32,
        device_ordinal: i32,
    ) -> HostServiceResultV1<BundleContextHandleV1>;
    fn release_context(&self, context: BundleContextHandleV1) -> HostServiceResultV1<()>;

    /// Selects the one pool a context has, under the same rule.
    fn select_memory_pool(
        &self,
        context: BundleContextHandleV1,
    ) -> HostServiceResultV1<BundleMemoryPoolHandleV1>;

    /// Takes or gives back one lease over a pool's grant to one peer context.
    ///
    /// `true` acquires a lease and `false` gives one back. The pair is a lease
    /// rather than a switch, because the device state behind it is shared: the
    /// host counts the leases one instance holds over each directed edge, asks
    /// the device only on the first acquire, and enables the grant only if it
    /// was not already there. The last release disables it only if this host
    /// enabled it, so a grant another user of the device made stays.
    ///
    /// A release with no lease behind it is
    /// [`BundleStatusV1::InvalidState`]. A refused enable or disable
    /// leaves the lease exactly as it was and is retryable. A pool a lease was
    /// granted through cannot be released while that lease stands.
    fn set_memory_pool_peer_access(
        &self,
        pool: BundleMemoryPoolHandleV1,
        peer_context: BundleContextHandleV1,
        enabled: bool,
    ) -> HostServiceResultV1<()>;
    fn release_memory_pool(&self, pool: BundleMemoryPoolHandleV1) -> HostServiceResultV1<()>;

    fn allocate_device_buffer(
        &self,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        byte_length: u64,
        alignment: u64,
    ) -> HostServiceResultV1<BundleDeviceBufferHandleV1>;
    fn release_device_buffer(&self, buffer: BundleDeviceBufferHandleV1) -> HostServiceResultV1<()>;

    fn open_compiled_image(
        &self,
        component_role: &[u8],
    ) -> HostServiceResultV1<BundleCompiledImageHandleV1>;
    fn upload_rank_arena(
        &self,
        context: BundleContextHandleV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        image: BundleCompiledImageHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1>;
    fn upload_rank_arenas(
        &self,
        uploads: &[RankArenaUploadV1],
    ) -> HostServiceResultV1<Vec<BundleRankArenaHandleV1>> {
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
                Err(status) => {
                    for arena in arenas.into_iter().rev() {
                        let _ = self.release_rank_arena(arena);
                    }
                    return Err(status);
                }
            }
        }
        Ok(arenas)
    }
    fn release_rank_arena(&self, arena: BundleRankArenaHandleV1) -> HostServiceResultV1<()>;
    fn resolve_compiled_image_view(
        &self,
        image: BundleCompiledImageHandleV1,
        arena: BundleRankArenaHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1>;
    fn release_compiled_image(&self, image: BundleCompiledImageHandleV1)
    -> HostServiceResultV1<()>;

    /// Copies rows of one host-mapped view of `gather.image` for
    /// `gather.rank` (`chop.host-mapped-gather.v1`) into `destination`,
    /// ordered on `stream` like [`Self::copy_host_to_device`]: row `i` of
    /// `destination` is row `gather.rows[i]` of the view, each
    /// `gather.row_bytes` long. The host reads the rows into page-locked
    /// memory of its own before it returns and keeps that memory until the
    /// copy completes. A view the image does not map for this rank is
    /// [`BundleStatusV1::NotFound`]; no rows, a row length that does not
    /// divide the view, a row past its end or a destination of any length
    /// but `rows.len() * row_bytes` is [`BundleStatusV1::InvalidArgument`].
    fn gather_mapped_rows(
        &self,
        _stream: BundleStreamHandleV1,
        _destination: DeviceBufferSpanV1,
        _gather: MappedRowGatherV1<'_>,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Register one authenticated row set without device or pinned staging.
    /// Caching is optional: an unavailable cache preserves uncached serving.
    fn register_mapped_row_set(&self, _planes: &[MappedRowPlaneV1]) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Allocate bounded reusable pinned staging for device-indexed mapped rows.
    /// Preparation and enqueue are outside capture. Inputs, outputs and their
    /// streams remain live until release. A failed completion must prevent
    /// publication of any results consuming this gather.
    fn create_mapped_gather(
        &self,
        _producer: BundleStreamHandleV1,
        _indices: DeviceBufferSpanV1,
        _planes: &[MappedGatherPlaneV1],
    ) -> HostServiceResultV1<crate::BundleMappedGatherHandleV1> {
        Err(BundleStatusV1::Unsupported)
    }
    /// Enqueue one ID D2H, host gather and event-ordered uploads without waiting.
    fn enqueue_mapped_gather(
        &self,
        _gather: crate::BundleMappedGatherHandleV1,
        _rows: u64,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    /// Insert the dependency on this consumer stream before its first use.
    fn wait_mapped_gather(
        &self,
        _gather: crate::BundleMappedGatherHandleV1,
        _consumer: BundleStreamHandleV1,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    /// Nonblocking completion, checked at the ordinary output boundary.
    fn complete_mapped_gather(
        &self,
        _gather: crate::BundleMappedGatherHandleV1,
    ) -> HostServiceResultV1<Option<MappedGatherStatsV1>> {
        Err(BundleStatusV1::Unsupported)
    }
    /// Release staging, fencing outstanding transfers on failure or cancellation.
    fn release_mapped_gather(
        &self,
        _gather: crate::BundleMappedGatherHandleV1,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn open_authenticated_component(
        &self,
        _component_role: &[u8],
    ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn inspect_authenticated_component(
        &self,
        _component: BundleAuthenticatedComponentHandleV1,
    ) -> HostServiceResultV1<AuthenticatedComponentMetadataV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn read_authenticated_component(
        &self,
        _component: BundleAuthenticatedComponentHandleV1,
        _role_destination: &mut [u8],
        _bytes_destination: &mut [u8],
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn release_authenticated_component(
        &self,
        _component: BundleAuthenticatedComponentHandleV1,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    /// Takes one lease over one directed context peer edge.
    ///
    /// Each handle is one lease. The host counts the leases one instance holds
    /// over each edge, enables the edge on the first one only if it was not
    /// enabled already, and remembers which of those happened. The last release
    /// disables it only if this host enabled it, so an edge another user of the
    /// devices made stays.
    ///
    /// A refused enable or disable leaves the lease exactly as it was and is
    /// retryable, and neither context of a live lease can be released while it
    /// stands.
    fn acquire_peer_access(
        &self,
        owner_context: BundleContextHandleV1,
        peer_context: BundleContextHandleV1,
    ) -> HostServiceResultV1<BundlePeerAccessHandleV1>;
    fn release_peer_access(&self, peer_access: BundlePeerAccessHandleV1)
    -> HostServiceResultV1<()>;

    fn create_stream(
        &self,
        context: BundleContextHandleV1,
        stream_kind: u32,
    ) -> HostServiceResultV1<BundleStreamHandleV1>;
    fn select_stage_stream(
        &self,
        context: BundleContextHandleV1,
        rank: u32,
        stage_id: &[u8],
        stream_kind: u32,
    ) -> HostServiceResultV1<BundleStreamHandleV1>;
    fn synchronize_stream(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()>;
    fn release_stream(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()>;

    fn create_event(
        &self,
        context: BundleContextHandleV1,
    ) -> HostServiceResultV1<BundleEventHandleV1>;
    fn record_event(
        &self,
        event: BundleEventHandleV1,
        stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<()>;
    fn wait_event(
        &self,
        stream: BundleStreamHandleV1,
        event: BundleEventHandleV1,
    ) -> HostServiceResultV1<()>;
    fn synchronize_event(&self, event: BundleEventHandleV1) -> HostServiceResultV1<()>;
    fn release_event(&self, event: BundleEventHandleV1) -> HostServiceResultV1<()>;

    fn copy_host_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()>;
    fn copy_device_to_host(
        &self,
        stream: BundleStreamHandleV1,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()>;
    fn copy_device_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()>;
    /// Copies `source`, owned by `source_context`, into `destination`, owned
    /// by `destination_context`, on `stream`. The stream belongs to either
    /// context: the destination's pulls, the source's pushes, and the
    /// stream's context must reach the other end's allocation through a
    /// directed peer lease and that allocation's pool grant. A host from
    /// before pushes refuses a source stream with `InvalidArgument` before
    /// submitting anything.
    fn copy_peer_to_peer(
        &self,
        stream: BundleStreamHandleV1,
        destination_context: BundleContextHandleV1,
        destination: DeviceBufferSpanV1,
        source_context: BundleContextHandleV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()>;
    fn zero_device_buffer(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()>;

    /// Writes a bounded table of resolved device addresses into one span.
    ///
    /// A bundle holds spans, not addresses, so a fixed kernel ABI that reads a
    /// table of device pointers out of device memory is something no copy can
    /// express. The bundle names the destination and the slots, in order, and
    /// the host resolves each slot, requires the destination to hold exactly
    /// `slots.len()` eight-byte entries, and writes them. `None` is the null
    /// pointer. The write completes before this returns.
    fn write_device_pointer_table(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        slots: &[Option<DeviceBufferSpanV1>],
    ) -> HostServiceResultV1<()>;

    fn load_cuda_module(
        &self,
        context: BundleContextHandleV1,
        module_bytes: &[u8],
    ) -> HostServiceResultV1<BundleCudaModuleHandleV1>;
    fn resolve_cuda_symbol(
        &self,
        module: BundleCudaModuleHandleV1,
        symbol_name: &[u8],
    ) -> HostServiceResultV1<BundleCudaSymbolHandleV1>;
    fn configure_cuda_symbol(
        &self,
        symbol: BundleCudaSymbolHandleV1,
        dynamic_shared_memory_bytes: u32,
    ) -> HostServiceResultV1<()>;
    fn launch_cuda_symbol(
        &self,
        stream: BundleStreamHandleV1,
        symbol: BundleCudaSymbolHandleV1,
        config: LaunchConfigV1,
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()>;
    fn release_cuda_symbol(&self, symbol: BundleCudaSymbolHandleV1) -> HostServiceResultV1<()>;
    fn release_cuda_module(&self, module: BundleCudaModuleHandleV1) -> HostServiceResultV1<()>;

    fn allocate_physical_kv(
        &self,
        request: &PhysicalKvRequestV1,
    ) -> HostServiceResultV1<BundlePhysicalKvHandleV1>;
    fn create_kv_request(
        &self,
        kv: BundlePhysicalKvHandleV1,
        quota_tokens: u64,
    ) -> HostServiceResultV1<BundleKvRequestHandleV1>;
    fn prepare_kv(
        &self,
        request: BundleKvRequestHandleV1,
        operation: &[u8],
    ) -> HostServiceResultV1<BundleKvTransactionHandleV1>;
    fn commit_kv(&self, transaction: BundleKvTransactionHandleV1) -> HostServiceResultV1<()>;
    fn cancel_kv(&self, transaction: BundleKvTransactionHandleV1) -> HostServiceResultV1<()>;
    fn resolve_kv_view(
        &self,
        request: BundleKvRequestHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<DeviceBufferSpanV1>;
    fn release_kv_request(&self, request: BundleKvRequestHandleV1) -> HostServiceResultV1<()>;
    fn release_physical_kv(&self, kv: BundlePhysicalKvHandleV1) -> HostServiceResultV1<()>;

    fn acquire_serving_kv(&self, _layout_id: &[u8]) -> HostServiceResultV1<u64> {
        Err(BundleStatusV1::Unsupported)
    }

    fn resolve_serving_kv_pool(&self, _rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
        Err(BundleStatusV1::Unsupported)
    }

    fn query_speculation(&self) -> HostServiceResultV1<crate::BundleRawSpeculationV1> {
        Ok(crate::BundleRawSpeculationV1::disabled())
    }

    fn query_batch_limits(&self) -> HostServiceResultV1<crate::BundleRawBatchLimitsV1> {
        Ok(crate::BundleRawBatchLimitsV1::unreported())
    }

    /// What the driver reports about keeping `symbol`'s clusters resident at
    /// `config`'s shape; a host that cannot tell reports nothing.
    fn query_cluster_residency(
        &self,
        _symbol: BundleCudaSymbolHandleV1,
        _config: LaunchConfigV1,
    ) -> HostServiceResultV1<crate::BundleRawClusterResidencyV1> {
        Ok(crate::BundleRawClusterResidencyV1::unreported())
    }

    fn allocate_state(&self, request: &StateRequestV1) -> HostServiceResultV1<BundleStateHandleV1>;
    fn resolve_state_view(
        &self,
        state: BundleStateHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1>;
    fn release_state(&self, state: BundleStateHandleV1) -> HostServiceResultV1<()>;

    fn create_collective(
        &self,
        request: &CollectiveRequestV1,
    ) -> HostServiceResultV1<BundleCollectiveHandleV1>;
    fn execute_collective(
        &self,
        collective: BundleCollectiveHandleV1,
        stream: BundleStreamHandleV1,
        operation: &[u8],
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()>;
    fn release_collective(&self, collective: BundleCollectiveHandleV1) -> HostServiceResultV1<()>;

    /// Runs one bundle-supplied operation under a context this host establishes.
    ///
    /// The host owes the whole of the boundary here: it resolves each handle,
    /// bounds checks each span against the allocation it names, refuses an
    /// operand set whose writable spans alias, waits on the stream, makes the
    /// context current, and lends the raw values for the length of one
    /// callback. A refusal invokes nothing, the wait included.
    ///
    /// The wait before the callback is unconditional, so an operation is never
    /// handed a stream with work it did not order still running. Whether the
    /// host waits again afterwards is `completion`'s to say: it does for an
    /// operation that may enqueue, and for a synchronous one there is nothing
    /// left to wait for.
    ///
    /// The answer is the operation's own code, which is zero for success. A
    /// nonzero code is the bundle's to interpret and travels back unchanged.
    fn run_custom_operation(
        &self,
        context: BundleContextHandleV1,
        stream: BundleStreamHandleV1,
        completion: CustomCompletionV1,
        operation_id: &[u8],
        operands: &[CustomOperandV1],
        operation: &CustomOperationV1<'_>,
    ) -> HostServiceResultV1<i32>;

    fn begin_graph_capture(&self, _stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    /// Ends the capture begun on `stream` and returns the graph it recorded.
    ///
    /// A failure once the capture ended (it was invalidated, or its graph
    /// failed to instantiate) leaves the stream out of capture, free to
    /// capture again. Only a failure before the capture ended leaves it
    /// capturing: ending it again is then the only way out, and every call
    /// a capture refuses stays refused until it is ended.
    fn end_graph_capture(
        &self,
        _stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<BundleGraphHandleV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn launch_graph(
        &self,
        _stream: BundleStreamHandleV1,
        _graph: BundleGraphHandleV1,
    ) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn release_graph(&self, _graph: BundleGraphHandleV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
}

/// A complete backend that reports every service as unsupported.
///
/// It supplies a structurally complete table where no service is callable yet,
/// which keeps table-shape checks independent of any host implementation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct UnsupportedHostServicesBackendV1;

macro_rules! unsupported_service_v1 {
    ($name:ident($($argument:ident: $type:ty),* $(,)?) -> $result:ty) => {
        fn $name(&self, $($argument: $type),*) -> HostServiceResultV1<$result> {
            Err(BundleStatusV1::Unsupported)
        }
    };
}

impl HostServicesBackendV1 for UnsupportedHostServicesBackendV1 {
    unsupported_service_v1!(log(_message: &[u8]) -> ());
    unsupported_service_v1!(
        select_context(_rank: u32, _device_ordinal: i32) -> BundleContextHandleV1
    );
    unsupported_service_v1!(release_context(_context: BundleContextHandleV1) -> ());
    unsupported_service_v1!(
        select_memory_pool(_context: BundleContextHandleV1) -> BundleMemoryPoolHandleV1
    );
    unsupported_service_v1!(
        set_memory_pool_peer_access(
            _pool: BundleMemoryPoolHandleV1,
            _peer_context: BundleContextHandleV1,
            _enabled: bool,
        ) -> ()
    );
    unsupported_service_v1!(release_memory_pool(_pool: BundleMemoryPoolHandleV1) -> ());
    unsupported_service_v1!(
        allocate_device_buffer(
            _pool: BundleMemoryPoolHandleV1,
            _stream: BundleStreamHandleV1,
            _byte_length: u64,
            _alignment: u64,
        ) -> BundleDeviceBufferHandleV1
    );
    unsupported_service_v1!(
        release_device_buffer(_buffer: BundleDeviceBufferHandleV1) -> ()
    );
    unsupported_service_v1!(
        open_compiled_image(_component_role: &[u8]) -> BundleCompiledImageHandleV1
    );
    unsupported_service_v1!(
        upload_rank_arena(
            _context: BundleContextHandleV1,
            _pool: BundleMemoryPoolHandleV1,
            _stream: BundleStreamHandleV1,
            _image: BundleCompiledImageHandleV1,
            _rank: u32,
        ) -> BundleRankArenaHandleV1
    );
    unsupported_service_v1!(release_rank_arena(_arena: BundleRankArenaHandleV1) -> ());
    unsupported_service_v1!(
        resolve_compiled_image_view(
            _image: BundleCompiledImageHandleV1,
            _arena: BundleRankArenaHandleV1,
            _rank: u32,
            _view_id: &[u8],
        ) -> CompiledImageSpanV1
    );
    unsupported_service_v1!(
        release_compiled_image(_image: BundleCompiledImageHandleV1) -> ()
    );
    unsupported_service_v1!(
        acquire_peer_access(
            _owner_context: BundleContextHandleV1,
            _peer_context: BundleContextHandleV1,
        ) -> BundlePeerAccessHandleV1
    );
    unsupported_service_v1!(
        release_peer_access(_peer_access: BundlePeerAccessHandleV1) -> ()
    );
    unsupported_service_v1!(
        create_stream(
            _context: BundleContextHandleV1,
            _stream_kind: u32,
        ) -> BundleStreamHandleV1
    );
    unsupported_service_v1!(
        select_stage_stream(
            _context: BundleContextHandleV1,
            _rank: u32,
            _stage_id: &[u8],
            _stream_kind: u32,
        ) -> BundleStreamHandleV1
    );
    unsupported_service_v1!(synchronize_stream(_stream: BundleStreamHandleV1) -> ());
    unsupported_service_v1!(release_stream(_stream: BundleStreamHandleV1) -> ());
    unsupported_service_v1!(
        create_event(_context: BundleContextHandleV1) -> BundleEventHandleV1
    );
    unsupported_service_v1!(
        record_event(
            _event: BundleEventHandleV1,
            _stream: BundleStreamHandleV1,
        ) -> ()
    );
    unsupported_service_v1!(
        wait_event(
            _stream: BundleStreamHandleV1,
            _event: BundleEventHandleV1,
        ) -> ()
    );
    unsupported_service_v1!(synchronize_event(_event: BundleEventHandleV1) -> ());
    unsupported_service_v1!(release_event(_event: BundleEventHandleV1) -> ());
    unsupported_service_v1!(
        copy_host_to_device(
            _stream: BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
            _source: &[u8],
        ) -> ()
    );
    unsupported_service_v1!(
        copy_device_to_host(
            _stream: BundleStreamHandleV1,
            _destination: &mut [u8],
            _source: DeviceBufferSpanV1,
        ) -> ()
    );
    unsupported_service_v1!(
        copy_device_to_device(
            _stream: BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
            _source: DeviceBufferSpanV1,
        ) -> ()
    );
    unsupported_service_v1!(
        copy_peer_to_peer(
            _stream: BundleStreamHandleV1,
            _destination_context: BundleContextHandleV1,
            _destination: DeviceBufferSpanV1,
            _source_context: BundleContextHandleV1,
            _source: DeviceBufferSpanV1,
        ) -> ()
    );
    unsupported_service_v1!(
        zero_device_buffer(
            _stream: BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
        ) -> ()
    );
    unsupported_service_v1!(
        write_device_pointer_table(
            _stream: BundleStreamHandleV1,
            _destination: DeviceBufferSpanV1,
            _slots: &[Option<DeviceBufferSpanV1>],
        ) -> ()
    );
    unsupported_service_v1!(
        load_cuda_module(
            _context: BundleContextHandleV1,
            _module_bytes: &[u8],
        ) -> BundleCudaModuleHandleV1
    );
    unsupported_service_v1!(
        resolve_cuda_symbol(
            _module: BundleCudaModuleHandleV1,
            _symbol_name: &[u8],
        ) -> BundleCudaSymbolHandleV1
    );
    unsupported_service_v1!(
        configure_cuda_symbol(
            _symbol: BundleCudaSymbolHandleV1,
            _dynamic_shared_memory_bytes: u32,
        ) -> ()
    );
    unsupported_service_v1!(
        launch_cuda_symbol(
            _stream: BundleStreamHandleV1,
            _symbol: BundleCudaSymbolHandleV1,
            _config: LaunchConfigV1,
            _arguments: &[KernelArgumentV1],
        ) -> ()
    );
    unsupported_service_v1!(release_cuda_symbol(_symbol: BundleCudaSymbolHandleV1) -> ());
    unsupported_service_v1!(release_cuda_module(_module: BundleCudaModuleHandleV1) -> ());
    unsupported_service_v1!(
        allocate_physical_kv(_request: &PhysicalKvRequestV1) -> BundlePhysicalKvHandleV1
    );
    unsupported_service_v1!(
        create_kv_request(
            _kv: BundlePhysicalKvHandleV1,
            _quota_tokens: u64,
        ) -> BundleKvRequestHandleV1
    );
    unsupported_service_v1!(
        prepare_kv(
            _request: BundleKvRequestHandleV1,
            _operation: &[u8],
        ) -> BundleKvTransactionHandleV1
    );
    unsupported_service_v1!(commit_kv(_transaction: BundleKvTransactionHandleV1) -> ());
    unsupported_service_v1!(cancel_kv(_transaction: BundleKvTransactionHandleV1) -> ());
    unsupported_service_v1!(
        resolve_kv_view(
            _request: BundleKvRequestHandleV1,
            _rank: u32,
            _view_id: &[u8],
        ) -> DeviceBufferSpanV1
    );
    unsupported_service_v1!(release_kv_request(_request: BundleKvRequestHandleV1) -> ());
    unsupported_service_v1!(release_physical_kv(_kv: BundlePhysicalKvHandleV1) -> ());
    unsupported_service_v1!(
        allocate_state(_request: &StateRequestV1) -> BundleStateHandleV1
    );
    unsupported_service_v1!(
        resolve_state_view(
            _state: BundleStateHandleV1,
            _rank: u32,
        ) -> DeviceBufferSpanV1
    );
    unsupported_service_v1!(release_state(_state: BundleStateHandleV1) -> ());
    unsupported_service_v1!(
        create_collective(_request: &CollectiveRequestV1) -> BundleCollectiveHandleV1
    );
    unsupported_service_v1!(
        execute_collective(
            _collective: BundleCollectiveHandleV1,
            _stream: BundleStreamHandleV1,
            _operation: &[u8],
            _arguments: &[KernelArgumentV1],
        ) -> ()
    );
    unsupported_service_v1!(
        release_collective(_collective: BundleCollectiveHandleV1) -> ()
    );
    unsupported_service_v1!(
        run_custom_operation(
            _context: BundleContextHandleV1,
            _stream: BundleStreamHandleV1,
            _completion: CustomCompletionV1,
            _operation_id: &[u8],
            _operands: &[CustomOperandV1],
            _operation: &CustomOperationV1<'_>,
        ) -> i32
    );
    unsupported_service_v1!(begin_graph_capture(_stream: BundleStreamHandleV1) -> ());
    unsupported_service_v1!(
        end_graph_capture(_stream: BundleStreamHandleV1) -> BundleGraphHandleV1
    );
    unsupported_service_v1!(
        launch_graph(_stream: BundleStreamHandleV1, _graph: BundleGraphHandleV1) -> ()
    );
    unsupported_service_v1!(release_graph(_graph: BundleGraphHandleV1) -> ());
}
