//! Optional instance-specific prefill scheduling declaration and execution.
//! Legacy tables, symbols, manifests and batch records remain unchanged.

use crate::{BundleAbiError, BundleExecuteFnV1, BundleHandleV1, BundleStatusV1};

pub const PREFILL_API_SYMBOL_V1: &[u8] = b"chop_bundle_prefill_api_v1\0";

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

#[cfg(test)]
mod tests {
    use super::*;

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
