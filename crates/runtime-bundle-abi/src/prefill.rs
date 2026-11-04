//! Optional instance-specific prefill scheduling declaration and execution.
//! Legacy tables, symbols, manifests and batch records remain unchanged.

use crate::{BundleAbiError, BundleExecuteFnV1, BundleHandleV1, BundleStatusV1};

pub const PREFILL_API_SYMBOL_V1: &[u8] = b"chop_bundle_prefill_api_v1\0";

/// Optional capability accepting scheduler-selected cuts before the next grid
/// boundary. It uses the existing table and envelope layouts through a separate
/// export, so hosts that only know the grid contract keep that contract.
/// Each admitted end promises complete state and results identical to the
/// uncut schedule, including another split within a restored final chunk.
pub const PREFILL_CUTS_API_SYMBOL_V1: &[u8] = b"chop_bundle_prefill_cuts_api_v1\0";

/// Zero chunk_tokens disables the contract. Otherwise the host schedules
/// unscored text requests to the next absolute multiple of chunk_tokens or
/// their prompt end, and captures prefill state only at those boundaries.
/// Decode/verify state remains complete. Unsupported requests keep legacy
/// full-row execution and receive no guaranteed-prefill envelope.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundlePrefillPolicyV1 {
    pub struct_size: u32,
    pub status: BundleStatusV1,
    pub chunk_tokens: u32,
    pub reserved: u32,
}

impl BundlePrefillPolicyV1 {
    pub const fn disabled() -> Self {
        Self::guaranteed(0)
    }

    pub const fn guaranteed(chunk_tokens: u32) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            status: BundleStatusV1::Success,
            chunk_tokens,
            reserved: 0,
        }
    }

    pub const fn failure(status: BundleStatusV1) -> Self {
        Self {
            status,
            ..Self::disabled()
        }
    }

    pub fn validate(self) -> Result<(), BundleAbiError> {
        if self.struct_size < std::mem::size_of::<Self>() as u32 || self.reserved != 0 {
            return Err(BundleAbiError::InvalidRange("prefill policy"));
        }
        crate::status_result(self.status)
    }
}

pub type BundleDescribePrefillFnV1 = extern "C" fn(BundleHandleV1) -> BundlePrefillPolicyV1;
/// The caller must retain the native module while calling its getter and
/// reading its immutable table. The export must provide a readable, aligned
/// table header; the loader checks its size and ABI before reading callbacks.
// SAFETY-REVIEWED: this declares the C getter contract; it executes no unsafe
// operation. The loader retains the module and validates the returned table.
pub type BundleGetPrefillApiFnV1 = unsafe extern "C" fn() -> *const BundlePrefillApiV1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundlePrefillApiV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub describe_prefill_v1: Option<BundleDescribePrefillFnV1>,
    pub execute_prefill_v1: Option<BundleExecuteFnV1>,
}

impl BundlePrefillApiV1 {
    pub fn validate(self) -> Result<(), BundleAbiError> {
        crate::validate_table_header(
            "prefill API",
            self.struct_size,
            std::mem::size_of::<Self>() as u32,
            self.abi_version,
        )?;
        if self.describe_prefill_v1.is_none() || self.execute_prefill_v1.is_none() {
            return Err(BundleAbiError::MissingApiCallback("prefill API"));
        }
        Ok(())
    }
}

/// Optional per-instance anchor-item allowance on a guaranteed grid with
/// cuts. An instance keeps the state of up to `anchor_items` prompt-ending
/// prefill items per step at the last prefill-granule multiple strictly
/// inside each (`BATCH_ITEM_FLAG_ANCHOR_V3`), so that anchor needs no chunk
/// end. This allowance applies only to guaranteed-grid requests, and only
/// in steps whose prefill items all carry a boundary in the cuts envelope;
/// a step with a media or other prefill item without one keeps every such
/// anchor at a chunk end. Manifest anchors are unchanged and go only to
/// items without a boundary, so a step keeps grid or manifest anchors,
/// never both.
/// Zero items keeps every such anchor at a chunk end. A nonzero
/// allowance requires an enabled cuts policy. Old hosts ignore the export,
/// and hosts use the manifest value for bundles without it.
pub const PREFILL_ANCHORS_API_SYMBOL_V2: &[u8] = b"chop_bundle_prefill_anchors_api_v2\0";

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundlePrefillAnchorsV2 {
    pub struct_size: u32,
    pub status: BundleStatusV1,
    pub anchor_items: u32,
}

impl BundlePrefillAnchorsV2 {
    pub const fn kept(anchor_items: u32) -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>() as u32,
            status: BundleStatusV1::Success,
            anchor_items,
        }
    }

    pub const fn failure(status: BundleStatusV1) -> Self {
        Self {
            status,
            ..Self::kept(0)
        }
    }

    pub fn validate(self) -> Result<(), BundleAbiError> {
        if self.struct_size < std::mem::size_of::<Self>() as u32 {
            return Err(BundleAbiError::InvalidRange("prefill anchors policy"));
        }
        crate::status_result(self.status)
    }
}

pub type BundleDescribePrefillAnchorsFnV2 = extern "C" fn(BundleHandleV1) -> BundlePrefillAnchorsV2;
/// The caller must retain the native module while calling its getter and
/// reading its immutable table, as for the prefill table.
// SAFETY-REVIEWED: this declares the C getter contract; it executes no unsafe
// operation. The loader retains the module and validates the returned table.
pub type BundleGetPrefillAnchorsApiFnV2 =
    unsafe extern "C" fn() -> *const BundlePrefillAnchorsApiV2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundlePrefillAnchorsApiV2 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub describe_prefill_anchors_v2: Option<BundleDescribePrefillAnchorsFnV2>,
}

impl BundlePrefillAnchorsApiV2 {
    pub fn validate(self) -> Result<(), BundleAbiError> {
        crate::validate_table_header(
            "prefill anchors API",
            self.struct_size,
            std::mem::size_of::<Self>() as u32,
            self.abi_version,
        )?;
        if self.describe_prefill_anchors_v2.is_none() {
            return Err(BundleAbiError::MissingApiCallback("prefill anchors API"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_policies_v2_validate_status_and_size() {
        assert_eq!(std::mem::size_of::<BundlePrefillAnchorsV2>(), 12);
        for items in [0, 1, 8] {
            let policy = BundlePrefillAnchorsV2::kept(items);
            policy.validate().unwrap();
            assert!(
                BundlePrefillAnchorsV2 {
                    struct_size: 11,
                    ..policy
                }
                .validate()
                .is_err()
            );
        }
        assert_eq!(
            BundlePrefillAnchorsV2::failure(BundleStatusV1::InvalidState).validate(),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
        );
    }

    #[test]
    fn policies_validate_status_size_and_reserved_fields() {
        assert_eq!(std::mem::size_of::<BundlePrefillPolicyV1>(), 16);
        for chunk in [0, 2048, 8192] {
            let policy = BundlePrefillPolicyV1::guaranteed(chunk);
            policy.validate().unwrap();
            assert!(
                BundlePrefillPolicyV1 {
                    reserved: 1,
                    ..policy
                }
                .validate()
                .is_err()
            );
            assert!(
                BundlePrefillPolicyV1 {
                    struct_size: 15,
                    ..policy
                }
                .validate()
                .is_err()
            );
        }
        assert_eq!(
            BundlePrefillPolicyV1::failure(BundleStatusV1::InvalidState).validate(),
            Err(BundleAbiError::Status(BundleStatusV1::InvalidState))
        );
    }
}
