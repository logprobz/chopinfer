//! Optional device-result export and its pointer-based result sink.
use super::*;
use crate::{
    BundleAbiError, BundleByteViewV1, BundleExecuteResultSinkV1, BundleHandleV1, BundleStatusV1,
    ExecuteResultSinkV1,
};
use std::ffi::c_void;

pub const DEVICE_RESULTS_API_SYMBOL_V1: &[u8] = b"chop_bundle_get_device_results_api_v1\0";

// SAFETY-REVIEWED: the execute owns user_data and lends a writable output record;
// returned metadata stays readable through that execute.
pub type BundleDescribeDeviceCycleFnV1 =
    unsafe extern "C" fn(*mut c_void, *mut BundleDeviceCycleViewV1) -> BundleStatusV1;
// SAFETY-REVIEWED: the enclosing execute owns the callback and user_data.
pub type BundleDeclareInputDependencyFnV1 =
    unsafe extern "C" fn(*mut c_void, u32) -> BundleStatusV1;
// SAFETY-REVIEWED: the execute owns user_data and lends the validated draft record.
pub type BundleEmitDeviceDraftsFnV1 =
    unsafe extern "C" fn(*mut c_void, *const BundleDeviceDraftsV1) -> BundleStatusV1;
// SAFETY-REVIEWED: the execute owns user_data and lends the validated sampling record.
pub type BundleSampleDeviceLogitsFnV1 =
    unsafe extern "C" fn(*mut c_void, *const BundleDeviceSamplingRequestV1) -> BundleStatusV1;
// SAFETY-REVIEWED: the execute owns user_data and lends the validated logits view.
pub type BundleEmitReadyLogitsFnV1 =
    unsafe extern "C" fn(*mut c_void, *const BundleReadyLogitsViewV1) -> BundleStatusV1;
// SAFETY-REVIEWED: the execute owns user_data and lends the validated completion record.
pub type BundleEmitDeviceCompletionFnV1 =
    unsafe extern "C" fn(*mut c_void, *const BundleDeviceCompletionV1) -> BundleStatusV1;

/// Passed by pointer only by the optional execute entry. The legacy
/// by-value sink and execute signature remain unchanged.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleDeviceResultSinkV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub base: BundleExecuteResultSinkV1,
    pub describe_device_cycle_v1: Option<BundleDescribeDeviceCycleFnV1>,
    pub declare_input_dependency_v1: Option<BundleDeclareInputDependencyFnV1>,
    pub emit_device_drafts_v1: Option<BundleEmitDeviceDraftsFnV1>,
    pub sample_device_logits_v1: Option<BundleSampleDeviceLogitsFnV1>,
    pub emit_ready_logits_view_v1: Option<BundleEmitReadyLogitsFnV1>,
    pub emit_device_completion_v1: Option<BundleEmitDeviceCompletionFnV1>,
}

impl BundleDeviceResultSinkV1 {
    pub(crate) fn validate(self) -> Result<(), BundleAbiError> {
        crate::validate_table_header(
            "device result sink",
            self.struct_size,
            std::mem::size_of::<Self>() as u32,
            self.abi_version,
        )?;
        crate::validate_execute_result_sink_v1(self.base)?;
        if self.describe_device_cycle_v1.is_none()
            || self.declare_input_dependency_v1.is_none()
            || self.emit_device_drafts_v1.is_none()
            || self.sample_device_logits_v1.is_none()
            || self.emit_ready_logits_view_v1.is_none()
            || self.emit_device_completion_v1.is_none()
        {
            return Err(BundleAbiError::Status(BundleStatusV1::Unsupported));
        }
        Ok(())
    }
}

// SAFETY-REVIEWED: the native caller lends the request, sink table and user_data
// for this call; the trampoline validates the table before invoking callbacks.
pub type BundleExecuteDeviceResultsFnV1 = unsafe extern "C" fn(
    BundleHandleV1,
    BundleByteViewV1,
    *const BundleDeviceResultSinkV1,
) -> BundleStatusV1;
// SAFETY-REVIEWED: the loader retains the module owning the immutable table and
// validates its size and ABI before reading the execute callback.
pub type BundleGetDeviceResultsApiFnV1 = unsafe extern "C" fn() -> *const BundleDeviceResultsApiV1;

/// Separate optional export, so legacy entry-table bytes also stay unchanged.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BundleDeviceResultsApiV1 {
    pub struct_size: u32,
    pub abi_version: u32,
    pub execute_device_results_v1: Option<BundleExecuteDeviceResultsFnV1>,
}

impl BundleDeviceResultsApiV1 {
    pub fn validate(self) -> Result<BundleExecuteDeviceResultsFnV1, BundleAbiError> {
        crate::validate_table_header(
            "device results API",
            self.struct_size,
            std::mem::size_of::<Self>() as u32,
            self.abi_version,
        )?;
        self.execute_device_results_v1
            .ok_or(BundleAbiError::MissingApiCallback(
                "execute_device_results_v1",
            ))
    }
}

impl ExecuteResultSinkV1 {
    #[must_use]
    pub const fn device_results_enabled(&self) -> bool {
        self.device_results.is_some()
    }

    pub(crate) fn with_device_results(
        mut self,
        table: BundleDeviceResultSinkV1,
    ) -> Result<Self, BundleAbiError> {
        table.validate()?;
        self.device_results = Some(table);
        Ok(self)
    }

    fn device_table(&self) -> Result<&BundleDeviceResultSinkV1, BundleAbiError> {
        self.device_results
            .as_ref()
            .ok_or(BundleAbiError::Status(BundleStatusV1::Unsupported))
    }

    /// Copies engine-owned metadata, available only during this execute.
    pub fn device_cycle_metadata(&self) -> Result<DeviceCycleMetadataV1, BundleAbiError> {
        crate::trampoline::device_results_describe(self.device_table()?, self.limits())
    }

    pub fn declare_input_dependency(&mut self, dependency: u32) -> Result<(), BundleAbiError> {
        if ![
            DEVICE_RESULTS_HOST_VALUES_V1,
            DEVICE_RESULTS_DEVICE_FEEDBACK_V1,
        ]
        .contains(&dependency)
        {
            return Err(BundleAbiError::Status(BundleStatusV1::Unsupported));
        }
        crate::trampoline::device_results_dependency(self.device_table()?, dependency)
    }

    pub fn emit_device_drafts(
        &mut self,
        drafts: BundleDeviceDraftsV1,
    ) -> Result<(), BundleAbiError> {
        drafts.validate().map_err(BundleAbiError::Status)?;
        crate::trampoline::device_results_drafts(self.device_table()?, drafts)
    }

    pub fn sample_device_logits(
        &mut self,
        request: BundleDeviceSamplingRequestV1,
    ) -> Result<(), BundleAbiError> {
        request.validate()?;
        let table = *self.device_table()?;
        crate::charge_result_capacity_rows_v1(
            self.limits(),
            &mut self.emitted_rows,
            &mut self.emitted_logits,
            u64::from(request.logits.logits.row_count),
            u64::from(request.logits.logits.live_width),
        )?;
        crate::trampoline::device_results_sample(&table, request)
    }

    pub fn emit_ready_logits_view(
        &mut self,
        view: BundleReadyLogitsViewV1,
    ) -> Result<(), BundleAbiError> {
        view.validate()?;
        let table = *self.device_table()?;
        crate::charge_result_capacity_rows_v1(
            self.limits(),
            &mut self.emitted_rows,
            &mut self.emitted_logits,
            u64::from(view.logits.row_count),
            u64::from(view.logits.live_width),
        )?;
        crate::trampoline::device_results_ready(&table, view)
    }

    pub fn emit_device_completion(
        &mut self,
        completion: BundleDeviceCompletionV1,
    ) -> Result<(), BundleAbiError> {
        completion.validate().map_err(BundleAbiError::Status)?;
        if u64::from(completion.sample_rows) > self.limits().maximum_rows {
            return Err(BundleAbiError::ExecuteResultCapacity);
        }
        crate::trampoline::device_results_completion(self.device_table()?, completion)
    }
}
