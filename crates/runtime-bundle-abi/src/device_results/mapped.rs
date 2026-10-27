//! Optional mapped logits input. Sampling and completion retain logical row order.
use super::*;
use crate::{
    BundleAbiError, BundleByteViewV1, BundleHandleV1, BundleStatusV1, ExecuteResultSinkV1,
};
use std::ffi::c_void;

pub const MAPPED_DEVICE_RESULTS_API_SYMBOL_V1: &[u8] =
    b"chop_bundle_get_mapped_device_results_api_v1\0";

/// Each logical sample reads row_indices[i] of source, while retaining the
/// host metadata at first_sample_row+i. Indices are device U32 values and
/// must be below source.logits.row_count. Invalid indices produce invalid
/// sample records, never an out-of-bounds read. Duplicate indices are allowed.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BundleMappedDeviceSamplingRequestV1 {
    pub struct_size: u32,
    pub reserved: u32,
    pub source: BundleReadyLogitsViewV1,
    pub first_sample_row: u32,
    pub sample_rows: u32,
    pub row_indices: crate::BundleDeviceBufferViewV1,
    pub output: crate::BundleDeviceBufferViewV1,
    pub done: crate::BundleEventHandleV1,
}

impl BundleMappedDeviceSamplingRequestV1 {
    pub fn validate(self) -> Result<(), BundleAbiError> {
        self.source.validate()?;
        crate::validate_device_buffer_view_v1(self.row_indices)?;
        crate::validate_device_buffer_view_v1(self.output)?;
        if self.struct_size < size_of::<Self>() as u32
            || self.reserved != 0
            || self.source.logits.first_item_index != 0
            || self.sample_rows == 0
            || self
                .first_sample_row
                .checked_add(self.sample_rows)
                .is_none()
            || !self.row_indices.buffer.has_generation()
            || self.row_indices.offset % 4 != 0
            || self.row_indices.length < u64::from(self.sample_rows) * 4
            || self
                .row_indices
                .offset
                .checked_add(self.row_indices.length)
                .is_none()
            || !self.output.buffer.has_generation()
            || self.output.offset % 8 != 0
            || self.output.length < u64::from(self.sample_rows) * DEVICE_SAMPLE_RECORD_BYTES_V1
            || self.output.offset.checked_add(self.output.length).is_none()
            || self.output.buffer == self.source.logits.buffer.buffer
            || self.output.buffer == self.row_indices.buffer
            || !self.done.has_generation()
            || self.done == self.source.ready
        {
            return Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument));
        }
        Ok(())
    }
}

// SAFETY-REVIEWED: the enclosing execute owns user_data and lends the request.
pub type BundleSampleMappedDeviceLogitsFnV1 =
    unsafe extern "C" fn(*mut c_void, *const BundleMappedDeviceSamplingRequestV1) -> BundleStatusV1;

/// A separate sink table. Existing device-result and legacy tables are unchanged.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleMappedDeviceResultSinkV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub base: BundleDeviceResultSinkV1,
    pub sample_mapped_device_logits_v1: Option<BundleSampleMappedDeviceLogitsFnV1>,
}
impl BundleMappedDeviceResultSinkV1 {
    pub(crate) fn validate(self) -> Result<(), BundleAbiError> {
        crate::validate_table_header(
            "mapped device result sink",
            self.struct_size,
            size_of::<Self>() as u32,
            self.abi_version,
        )?;
        self.base.validate()
    }
}

// SAFETY-REVIEWED: the native execute lends its request, table and user_data.
pub type BundleExecuteMappedDeviceResultsFnV1 = unsafe extern "C" fn(
    BundleHandleV1,
    BundleByteViewV1,
    *const BundleMappedDeviceResultSinkV1,
) -> BundleStatusV1;
// SAFETY-REVIEWED: the loaded module owns the immutable returned table.
pub type BundleGetMappedDeviceResultsApiFnV1 =
    unsafe extern "C" fn() -> *const BundleMappedDeviceResultsApiV1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleMappedDeviceResultsApiV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub execute_mapped_device_results_v1: Option<BundleExecuteMappedDeviceResultsFnV1>,
}
impl BundleMappedDeviceResultsApiV1 {
    pub fn validate(self) -> Result<BundleExecuteMappedDeviceResultsFnV1, BundleAbiError> {
        crate::validate_table_header(
            "mapped device results API",
            self.struct_size,
            size_of::<Self>() as u32,
            self.abi_version,
        )?;
        self.execute_mapped_device_results_v1
            .ok_or(BundleAbiError::MissingApiCallback(
                "execute_mapped_device_results_v1",
            ))
    }
}

impl ExecuteResultSinkV1 {
    #[must_use]
    pub fn mapped_sampling_enabled(&self) -> bool {
        self.mapped_device_results
            .is_some_and(|table| table.sample_mapped_device_logits_v1.is_some())
    }
    pub fn sample_mapped_device_logits(
        &mut self,
        request: BundleMappedDeviceSamplingRequestV1,
    ) -> Result<(), BundleAbiError> {
        request.validate()?;
        let table = self
            .mapped_device_results
            .ok_or(BundleAbiError::Status(BundleStatusV1::Unsupported))?;
        crate::charge_result_capacity_rows_v1(
            self.limits(),
            &mut self.emitted_rows,
            &mut self.emitted_logits,
            u64::from(request.sample_rows),
            u64::from(request.source.logits.live_width),
        )?;
        crate::trampoline::device_results_sample_mapped(&table, request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BUNDLE_LOGITS_ELEMENT_V1_F32, BundleContextHandleV1, BundleDeviceBufferHandleV1,
        BundleDeviceBufferViewV1, BundleEventHandleV1, BundleLogitsViewV1,
    };
    fn request() -> BundleMappedDeviceSamplingRequestV1 {
        let buffer = |slot, bytes| BundleDeviceBufferViewV1 {
            buffer: BundleDeviceBufferHandleV1::from_parts(slot, 1),
            length: bytes,
            ..BundleDeviceBufferViewV1::empty()
        };
        BundleMappedDeviceSamplingRequestV1 {
            struct_size: size_of::<BundleMappedDeviceSamplingRequestV1>() as u32,
            reserved: 0,
            source: BundleReadyLogitsViewV1 {
                struct_size: size_of::<BundleReadyLogitsViewV1>() as u32,
                flags: 0,
                cycle_epoch: 1,
                logits: BundleLogitsViewV1 {
                    context: BundleContextHandleV1::from_parts(1, 1),
                    element_type: BUNDLE_LOGITS_ELEMENT_V1_F32,
                    buffer: buffer(1, 32),
                    row_count: 2,
                    live_width: 4,
                    row_stride_elements: 4,
                    ..BundleLogitsViewV1::empty()
                },
                ready: BundleEventHandleV1::from_parts(1, 1),
            },
            first_sample_row: 0,
            sample_rows: 3,
            row_indices: buffer(2, 12),
            output: buffer(3, 768),
            done: BundleEventHandleV1::from_parts(2, 1),
        }
    }
    #[test]
    fn mapped_input_bounds_physical_and_logical_rows_separately() {
        let valid = request();
        valid.validate().unwrap();
        for mutate in [
            |r: &mut BundleMappedDeviceSamplingRequestV1| r.struct_size = 4,
            |r: &mut BundleMappedDeviceSamplingRequestV1| r.sample_rows = 0,
            |r: &mut BundleMappedDeviceSamplingRequestV1| r.first_sample_row = u32::MAX,
            |r: &mut BundleMappedDeviceSamplingRequestV1| r.row_indices.length -= 1,
            |r: &mut BundleMappedDeviceSamplingRequestV1| r.row_indices.offset = 1,
            |r: &mut BundleMappedDeviceSamplingRequestV1| r.source.logits.buffer.length -= 1,
            |r: &mut BundleMappedDeviceSamplingRequestV1| r.output.length -= 1,
            |r: &mut BundleMappedDeviceSamplingRequestV1| r.output.buffer = r.row_indices.buffer,
            |r: &mut BundleMappedDeviceSamplingRequestV1| r.done = r.source.ready,
        ] {
            let mut bad = valid;
            mutate(&mut bad);
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }
}
