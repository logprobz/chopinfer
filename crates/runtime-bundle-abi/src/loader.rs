//! Process-lifetime dynamic loader for the one V1 bundle entry point.

// SAFETY-REVIEWED: this module contains the sole reviewed dynamic-loader boundary.
#![allow(unsafe_code)]

use std::fmt;
use std::sync::Arc;

#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::ffi::{CStr, CString, c_void};
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::os::unix::ffi::OsStrExt;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::path::Path;

use crate::{
    BundleAbiError, BundleApiHeaderV1, BundleApiV1, BundleApiWithPrefixCacheV1,
    BundleApiWithPrefixStateSlotV1, BundleApiWithRequestRetirementV1, BundleApiWithSpeculationV1,
    BundleCommitAcceptedFnV1, BundleDescribePrefixStateFnV1, BundleGetApiV1Fn, BundleHandleV1,
    BundleInstanceV1, BundleKeptRowsV1, BundlePrefixStateFnV1, BundleRawApiV1,
    BundleRestorePrefixStateInSlotFnV1, BundleRetireRequestFnV1, COMMIT_ACCEPTED_END_V1,
    HostServicesTableV1, LogitsRowSinkV1, PREFIX_CACHE_END_V1, PREFIX_STATE_SLOT_END_V1,
    PrefixCacheCallbacksV1, REQUEST_RETIREMENT_END_V1, validate_bundle_api_header_v1,
};

#[cfg(any(target_os = "linux", target_os = "macos"))]
const ENTRY_SYMBOL_V1: &[u8] = b"chop_bundle_get_api_v1\0";

trait ApiTableReaderV1 {
    fn read_header(&self) -> BundleApiHeaderV1;

    fn read_supported_table(&self) -> BundleRawApiV1;

    fn read_retirement_callback(&self) -> Option<BundleRetireRequestFnV1> {
        None
    }

    fn read_commit_accepted_callback(&self) -> Option<BundleCommitAcceptedFnV1> {
        None
    }

    fn read_prefix_cache_callbacks(&self) -> Option<PrefixCacheCallbacksV1> {
        None
    }

    fn read_prefix_state_slot_callback(&self) -> Option<BundleRestorePrefixStateInSlotFnV1> {
        None
    }
}

fn checked_api_from_reader(reader: &impl ApiTableReaderV1) -> Result<BundleApiV1, BundleAbiError> {
    let header = reader.read_header();
    validate_bundle_api_header_v1(header)?;
    let base = reader.read_supported_table();
    let mut api = BundleApiV1::new(base)?;
    if header.struct_size as usize >= REQUEST_RETIREMENT_END_V1 {
        api.retire_request_v1 = reader.read_retirement_callback();
    }
    if header.struct_size as usize >= COMMIT_ACCEPTED_END_V1 {
        api.commit_accepted_v1 = reader.read_commit_accepted_callback();
    }
    if header.struct_size as usize >= PREFIX_CACHE_END_V1 {
        api.prefix_cache_v1 = reader.read_prefix_cache_callbacks();
    }
    if header.struct_size as usize >= PREFIX_STATE_SLOT_END_V1 && api.prefix_cache_v1.is_some() {
        api.restore_prefix_state_in_slot_v1 = reader.read_prefix_state_slot_callback();
    }
    Ok(api)
}

struct NativeApiTableReaderV1 {
    pointer: *const BundleRawApiV1,
}

impl NativeApiTableReaderV1 {
    /// Reads one tail slot of the prefix-cache table.
    ///
    /// # Safety
    /// The validated table size covers the slot at `offset`, which holds a `T`.
    // SAFETY-REVIEWED: callers pass only offsets inside the validated table size.
    unsafe fn read_tail<T>(&self, offset: usize) -> T {
        // SAFETY-REVIEWED: the caller proved the validated size covers this slot.
        unsafe { std::ptr::read(self.pointer.cast::<u8>().add(offset).cast::<T>()) }
    }
}

impl ApiTableReaderV1 for NativeApiTableReaderV1 {
    fn read_prefix_cache_callbacks(&self) -> Option<PrefixCacheCallbacksV1> {
        // SAFETY-REVIEWED: the caller read these only after the size check covering all three.
        let (describe, capture, restore) = unsafe {
            (
                self.read_tail::<Option<BundleDescribePrefixStateFnV1>>(std::mem::offset_of!(
                    BundleApiWithPrefixCacheV1,
                    describe_prefix_state_v1
                )),
                self.read_tail::<Option<BundlePrefixStateFnV1>>(std::mem::offset_of!(
                    BundleApiWithPrefixCacheV1,
                    capture_prefix_state_v1
                )),
                self.read_tail::<Option<BundlePrefixStateFnV1>>(std::mem::offset_of!(
                    BundleApiWithPrefixCacheV1,
                    restore_prefix_state_v1
                )),
            )
        };
        PrefixCacheCallbacksV1::from_table(describe, capture, restore)
    }

    fn read_prefix_state_slot_callback(&self) -> Option<BundleRestorePrefixStateInSlotFnV1> {
        // SAFETY-REVIEWED: the caller read this only after the size check covering the slot.
        unsafe {
            self.read_tail::<Option<BundleRestorePrefixStateInSlotFnV1>>(std::mem::offset_of!(
                BundleApiWithPrefixStateSlotV1,
                restore_prefix_state_in_slot_v1
            ))
        }
    }

    fn read_commit_accepted_callback(&self) -> Option<BundleCommitAcceptedFnV1> {
        // SAFETY-REVIEWED: the validated size covers only this fixed tail slot.
        unsafe {
            std::ptr::read(
                self.pointer
                    .cast::<u8>()
                    .add(std::mem::offset_of!(
                        BundleApiWithSpeculationV1,
                        commit_accepted_v1
                    ))
                    .cast::<Option<BundleCommitAcceptedFnV1>>(),
            )
        }
    }
    fn read_retirement_callback(&self) -> Option<BundleRetireRequestFnV1> {
        // SAFETY-REVIEWED: the validated size covers this entire field. Read only
        // the tail slot, without forming a reference to an extended allocation.
        unsafe {
            std::ptr::read(
                self.pointer
                    .cast::<u8>()
                    .add(std::mem::offset_of!(
                        BundleApiWithRequestRetirementV1,
                        retire_request_v1
                    ))
                    .cast::<Option<BundleRetireRequestFnV1>>(),
            )
        }
    }

    fn read_header(&self) -> BundleApiHeaderV1 {
        // SAFETY-REVIEWED: a non-null V1 table pointer exposes its fixed aligned header.
        unsafe { std::ptr::read(self.pointer.cast::<BundleApiHeaderV1>()) }
    }

    fn read_supported_table(&self) -> BundleRawApiV1 {
        // SAFETY-REVIEWED: the validated advertised size covers the full supported V1 table.
        unsafe { std::ptr::read(self.pointer) }
    }
}

#[derive(Debug)]
pub enum BundleLoaderError {
    PathContainsNul,
    StageImage { path: String, message: String },
    Open { path: String, message: String },
    Symbol { path: String, message: String },
    NullApi { path: String },
    InvalidApi(BundleAbiError),
    UnsupportedPlatform,
}

impl fmt::Display for BundleLoaderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PathContainsNul => formatter.write_str("bundle library path contains a NUL byte"),
            Self::StageImage { path, message } => write!(
                formatter,
                "cannot stage a private loader image for bundle library '{path}': {message}"
            ),
            Self::Open { path, message } => {
                write!(formatter, "cannot open bundle library '{path}': {message}")
            }
            Self::Symbol { path, message } => write!(
                formatter,
                "bundle library '{path}' does not export chop_bundle_get_api_v1: {message}"
            ),
            Self::NullApi { path } => write!(
                formatter,
                "bundle library '{path}' returned a null V1 API pointer"
            ),
            Self::InvalidApi(error) => error.fmt(formatter),
            Self::UnsupportedPlatform => formatter.write_str(
                "runtime bundle dynamic loading requires a Linux or macOS host that can hold one \
                 private loader image",
            ),
        }
    }
}

impl std::error::Error for BundleLoaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidApi(error) => Some(error),
            Self::PathContainsNul
            | Self::StageImage { .. }
            | Self::Open { .. }
            | Self::Symbol { .. }
            | Self::NullApi { .. }
            | Self::UnsupportedPlatform => None,
        }
    }
}

/// An open module. Instances and frontend handles hold it in an `Arc`, so
/// the module stays mapped until the last of them is gone.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) struct DynamicLibraryV1 {
    handle: *mut c_void,
    // Drop runs dlclose before fields drop. Keep the loader path and its file
    // identity alive through unload, including process-lifetime poison retention.
    image: PrivateModuleImageV1,
}

// SAFETY-REVIEWED: a dlopen handle may be used and closed from any thread; after open
// only Drop reads it, once, and the staged image is plain owned files and paths.
#[cfg(any(target_os = "linux", target_os = "macos"))]
unsafe impl Send for DynamicLibraryV1 {}
// SAFETY-REVIEWED: as above; a shared reference reads nothing through the handle.
#[cfg(any(target_os = "linux", target_os = "macos"))]
unsafe impl Sync for DynamicLibraryV1 {}

#[cfg(any(target_os = "linux", target_os = "macos"))]
impl Drop for DynamicLibraryV1 {
    fn drop(&mut self) {
        // SAFETY-REVIEWED: this object uniquely owns a successful dlopen handle.
        let _ = unsafe { libc::dlclose(self.handle) };
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) struct DynamicLibraryV1;

/// A loaded V1 table whose library remains open for every created instance.
pub struct BundleLibraryV1 {
    api: BundleApiV1,
    library: Arc<DynamicLibraryV1>,
    /// The optional direct-loading exports (`weight_plan`).
    weight_plan: Option<crate::weight_plan::BundleWeightPlanFnV1>,
    weight_transform: Option<crate::weight_plan::BundleWeightTransformFnV1>,
    weight_census: Option<crate::weight_plan::BundleWeightPlanFnV1>,
    /// The optional chat frontend export.
    frontend: Option<crate::BundleFrontendApiFnV1>,
    device_results: Option<crate::device_results::BundleGetDeviceResultsApiFnV1>,
    prefill: Option<crate::prefill::BundleGetPrefillApiFnV1>,
}

impl BundleLibraryV1 {
    /// Loads the exact module bytes a caller already authenticated.
    ///
    /// Handing the loader a package path, or even the open file that verified,
    /// still leaves it mapping an inode other writers hold: a replaced
    /// directory entry redirects the first, and a write through the same inode
    /// defeats the second. The authenticated vector is therefore copied into a
    /// private image no package writer can name or change, and that copy is
    /// what the loader maps. `display` is used only in diagnostics.
    ///
    /// A platform that cannot hold such an image is refused rather than quietly
    /// falling back to a second lookup of the package path. A platform that can
    /// only give the copy a name has to show, once the load is over, that the
    /// name still held it, so this can fail after the module is already open.
    pub fn open_verified_image(bytes: &[u8], display: &str) -> Result<Self, BundleLoaderError> {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            let image = PrivateModuleImageV1::stage(bytes, display)?;
            let library = Self::open_unix(image, display.to_owned())?;
            // Whatever the platform could not make impossible, it answers for
            // here: a load that mapped anything but this copy ends in an error
            // rather than in a library.
            library.library.image.confirm_loaded(bytes, display)?;
            Ok(library)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            let _ = (bytes, display);
            Err(BundleLoaderError::UnsupportedPlatform)
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    fn open_unix(image: PrivateModuleImageV1, display: String) -> Result<Self, BundleLoaderError> {
        let path = CString::new(image.path().as_os_str().as_bytes())
            .map_err(|_| BundleLoaderError::PathContainsNul)?;
        // SAFETY-REVIEWED: dlopen receives a NUL-terminated path and fixed local eager-load flags.
        let handle = unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_LOCAL | libc::RTLD_NOW) };
        if handle.is_null() {
            return Err(BundleLoaderError::Open {
                path: display,
                message: dl_error_message(),
            });
        }
        let library = Arc::new(DynamicLibraryV1 { handle, image });
        let entry = CStr::from_bytes_with_nul(ENTRY_SYMBOL_V1)
            .expect("fixed bundle entry symbol is NUL-terminated");
        // SAFETY-REVIEWED: dlerror clears thread-local loader state before the paired dlsym call.
        let _ = unsafe { libc::dlerror() };
        // SAFETY-REVIEWED: dlsym receives an owned live library handle and a static symbol name.
        let symbol = unsafe { libc::dlsym(handle, entry.as_ptr()) };
        if symbol.is_null() {
            return Err(BundleLoaderError::Symbol {
                path: display,
                message: dl_error_message(),
            });
        }
        // SAFETY-REVIEWED: dlsym resolved the exact V1 entry symbol with the declared C signature.
        let get_api: BundleGetApiV1Fn = unsafe { std::mem::transmute(symbol) };
        // SAFETY-REVIEWED: the resolved V1 entry takes no argument and returns its own static table.
        let api_pointer = unsafe { get_api() };
        if api_pointer.is_null() {
            return Err(BundleLoaderError::NullApi { path: display });
        }
        let api = checked_api_from_reader(&NativeApiTableReaderV1 {
            pointer: api_pointer,
        })
        .map_err(BundleLoaderError::InvalidApi)?;
        let optional = |name: &[u8]| {
            let name = CStr::from_bytes_with_nul(name).expect("export names are NUL-terminated");
            // SAFETY-REVIEWED: dlsym receives the live library handle and a static symbol name;
            // a missing optional export is null.
            let symbol = unsafe { libc::dlsym(library.handle, name.as_ptr()) };
            (!symbol.is_null()).then_some(symbol)
        };
        // SAFETY-REVIEWED: each optional export is resolved by its exact V1 name,
        // whose C signature the header declares.
        let weight_plan =
            optional(crate::weight_plan::WEIGHT_PLAN_SYMBOL_V1).map(|symbol| unsafe {
                std::mem::transmute::<*mut c_void, crate::weight_plan::BundleWeightPlanFnV1>(symbol)
            });
        // SAFETY-REVIEWED: as above, for the bundle transform export.
        let weight_transform =
            optional(crate::weight_plan::WEIGHT_TRANSFORM_SYMBOL_V1).map(|symbol| unsafe {
                std::mem::transmute::<*mut c_void, crate::weight_plan::BundleWeightTransformFnV1>(
                    symbol,
                )
            });
        // SAFETY-REVIEWED: as above, for the census export, which has the plan
        // export's signature.
        let weight_census =
            optional(crate::weight_plan::WEIGHT_CENSUS_SYMBOL_V1).map(|symbol| unsafe {
                std::mem::transmute::<*mut c_void, crate::weight_plan::BundleWeightPlanFnV1>(symbol)
            });
        // SAFETY-REVIEWED: as above, for the frontend table export.
        let frontend = optional(crate::frontend::FRONTEND_API_SYMBOL_V1).map(|symbol| unsafe {
            std::mem::transmute::<*mut c_void, crate::BundleFrontendApiFnV1>(symbol)
        });
        // SAFETY-REVIEWED: the optional export is resolved by its exact name and C signature.
        let device_results =
            optional(crate::device_results::DEVICE_RESULTS_API_SYMBOL_V1).map(|symbol| unsafe {
                std::mem::transmute::<
                    *mut c_void,
                    crate::device_results::BundleGetDeviceResultsApiFnV1,
                >(symbol)
            });
        // SAFETY-REVIEWED: this exact optional export has the V1 C getter signature.
        let prefill = optional(crate::prefill::PREFILL_API_SYMBOL_V1).map(|symbol| unsafe {
            std::mem::transmute::<*mut c_void, crate::prefill::BundleGetPrefillApiFnV1>(symbol)
        });
        Ok(Self {
            api,
            library,
            weight_plan,
            weight_transform,
            weight_census,
            frontend,
            device_results,
            prefill,
        })
    }

    /// Whether the bundle exports `chop_bundle_weight_census_v1`.
    #[must_use]
    pub const fn has_weight_census(&self) -> bool {
        self.weight_census.is_some()
    }

    /// Asks the bundle what to measure before it plans `request`'s image and
    /// rank: the census and its encoded bytes, whose SHA-256 the plan request
    /// then states, or `None` when it measures nothing.
    pub fn weight_census(
        &self,
        request: &crate::weight_plan::WeightPlanRequestV1,
    ) -> Result<Option<(crate::weight_plan::WeightCensusV1, Vec<u8>)>, String> {
        let census = self
            .weight_census
            .ok_or("the bundle does not export chop_bundle_weight_census_v1")?;
        let answer = crate::trampoline::call_weight_census_v1(census, request)?;
        if let Some((census, _)) = &answer
            && (census.ranks, census.rank) != (request.ranks, request.rank)
        {
            return Err("the bundle measured another rank".into());
        }
        Ok(answer)
    }

    /// Asks the bundle for one image's and rank's weight plan, `None` when it
    /// declines the image. The answer is decoded and checked for its own
    /// consistency; its records are not yet executed.
    pub fn weight_plan(
        &self,
        request: &crate::weight_plan::WeightPlanRequestV1,
    ) -> Result<Option<crate::weight_plan::WeightPlanV1>, String> {
        let plan = self
            .weight_plan
            .ok_or("the bundle does not export chop_bundle_weight_plan_v1")?;
        let Some(answer) = crate::trampoline::call_weight_plan_v1(plan, request)? else {
            return Ok(None);
        };
        if (answer.ranks, answer.rank) != (request.ranks, request.rank)
            || (!request.image_role.is_empty() && answer.image_role != request.image_role)
        {
            return Err("the bundle planned another image or rank".into());
        }
        Ok(Some(answer))
    }

    /// The bundle's transform export, callable from any thread through
    /// [`crate::call_weight_transform_v1`] while this library is loaded.
    #[must_use]
    pub const fn weight_transform(&self) -> Option<crate::weight_plan::BundleWeightTransformFnV1> {
        self.weight_transform
    }

    /// The bundle's chat frontend (`chop_bundle_frontend_api_v1`), its table
    /// checked, or `None` when the bundle exports none. The export and every
    /// handle opened from it hold this module, so it stays mapped until the
    /// last of them closes, even after this library is dropped.
    pub fn frontend_v1(&self) -> Result<Option<crate::frontend::FrontendExportV1>, BundleAbiError> {
        self.frontend
            .map(|api| {
                // SAFETY-REVIEWED: `api` is this library's own export, resolved by its V1 name
                // and signature, and the export holds the library so its code stays loaded.
                unsafe {
                    crate::frontend::FrontendExportV1::from_module(
                        api,
                        Some(Arc::clone(&self.library)),
                    )
                }
            })
            .transpose()
    }

    /// Reports actual callback presence before creating an instance or GPU state.
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

    #[must_use]
    pub const fn has_device_results_export(&self) -> bool {
        self.device_results.is_some()
    }

    /// Validate the optional export only after the caller admits its manifest capability.
    pub fn validate_device_results_api_v1(
        &self,
    ) -> Result<crate::device_results::BundleDeviceResultsApiV1, BundleAbiError> {
        let getter = self
            .device_results
            .ok_or(BundleAbiError::MissingApiCallback("device results export"))?;
        // SAFETY-REVIEWED: the loaded module owns this immutable table for the library lifetime.
        let pointer = unsafe { getter() };
        if pointer.is_null() {
            return Err(BundleAbiError::InvalidHandle);
        }
        // SAFETY-REVIEWED: a native table provides its header before its optional tail is read.
        let header = unsafe { pointer.cast::<BundleApiHeaderV1>().read() };
        crate::validate_table_header(
            "device results API",
            header.struct_size,
            std::mem::size_of::<crate::device_results::BundleDeviceResultsApiV1>() as u32,
            header.abi_version,
        )?;
        // SAFETY-REVIEWED: the validated header covers the known table prefix.
        let api = unsafe { pointer.read() };
        api.validate()?;
        Ok(api)
    }

    /// Create an explicitly opted-in instance. Ordinary create does not invoke
    /// or enable the separate export, even when the module offers it.
    pub fn create_device_results_v1(
        &self,
        host_services: HostServicesTableV1,
        configuration: &[u8],
    ) -> Result<LoadedBundleInstanceV1, BundleAbiError> {
        let api = self
            .api_with_prefill()?
            .with_device_results_api(self.validate_device_results_api_v1()?)?;
        Ok(LoadedBundleInstanceV1 {
            instance: api.create(host_services, configuration)?,
            library: Arc::clone(&self.library),
        })
    }

    pub fn create(
        &self,
        host_services: HostServicesTableV1,
        configuration: &[u8],
    ) -> Result<LoadedBundleInstanceV1, BundleAbiError> {
        let instance = self
            .api_with_prefill()?
            .create(host_services, configuration)?;
        Ok(LoadedBundleInstanceV1 {
            instance,
            library: Arc::clone(&self.library),
        })
    }

    fn api_with_prefill(&self) -> Result<BundleApiV1, BundleAbiError> {
        let Some(getter) = self.prefill else {
            return Ok(self.api);
        };
        // SAFETY-REVIEWED: the loaded native module owns the returned immutable table.
        let pointer = unsafe { getter() };
        if pointer.is_null() {
            return Err(BundleAbiError::InvalidHandle);
        }
        // SAFETY-REVIEWED: only the fixed table header is read before its size is checked.
        let header = unsafe { pointer.cast::<BundleApiHeaderV1>().read() };
        crate::validate_table_header(
            "prefill API",
            header.struct_size,
            std::mem::size_of::<crate::prefill::BundlePrefillApiV1>() as u32,
            header.abi_version,
        )?;
        // SAFETY-REVIEWED: the header covers the complete known table prefix.
        self.api.with_prefill_api(unsafe { pointer.read() })
    }
}

/// A bundle instance that keeps its shared object loaded through destruction.
///
/// If a terminal lifecycle callback fails, the instance is poisoned. Dropping
/// it pins the shared object for process lifetime because native cleanup is no
/// longer safe to call.
pub struct LoadedBundleInstanceV1 {
    // Fields drop in declaration order, so the instance destroys before dlclose.
    instance: BundleInstanceV1,
    library: Arc<DynamicLibraryV1>,
}

impl LoadedBundleInstanceV1 {
    pub fn prefill_policy(&self) -> Result<crate::prefill::BundlePrefillPolicyV1, BundleAbiError> {
        self.instance.prefill_policy()
    }

    pub fn execute_prefill<S: LogitsRowSinkV1>(
        &mut self,
        request: &[u8],
        sink: &mut S,
    ) -> Result<(), BundleAbiError> {
        self.instance.execute_prefill(request, sink)
    }
    #[must_use]
    pub const fn supports_device_results(&self) -> bool {
        self.instance.supports_device_results()
    }

    #[must_use]
    pub fn last_execute_refusal(&self) -> Option<&str> {
        self.instance.last_execute_refusal()
    }

    #[must_use]
    pub const fn supports_speculation(&self) -> bool {
        self.instance.supports_speculation()
    }
    #[must_use]
    pub const fn supports_request_retirement(&self) -> bool {
        self.instance.supports_request_retirement()
    }

    pub fn retire_request(&mut self, request_id: u64) -> Result<(), BundleAbiError> {
        self.instance.retire_request(request_id)
    }

    #[must_use]
    pub const fn supports_prefix_cache(&self) -> bool {
        self.instance.supports_prefix_cache()
    }

    #[must_use]
    pub const fn supports_prefix_state_slot(&self) -> bool {
        self.instance.supports_prefix_state_slot()
    }

    pub fn describe_prefix_state(&mut self) -> Result<u64, BundleAbiError> {
        self.instance.describe_prefix_state()
    }

    pub fn capture_prefix_state(
        &mut self,
        request_id: u64,
        tokens: u64,
        pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        self.instance
            .capture_prefix_state(request_id, tokens, pages)
    }

    pub fn restore_prefix_state(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
        pages: &[u64],
    ) -> Result<(), BundleAbiError> {
        self.instance
            .restore_prefix_state(request_id, token_ids, pages)
    }

    pub fn restore_prefix_state_in_slot(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
        pages: &[u64],
        state_slot: u64,
    ) -> Result<(), BundleAbiError> {
        self.instance
            .restore_prefix_state_in_slot(request_id, token_ids, pages, state_slot)
    }

    #[must_use]
    pub const fn handle(&self) -> BundleHandleV1 {
        self.instance.handle()
    }

    pub fn execute<S: LogitsRowSinkV1>(
        &mut self,
        request: &[u8],
        result_sink: &mut S,
    ) -> Result<(), BundleAbiError> {
        self.instance.execute(request, result_sink)
    }

    pub fn commit(&mut self) -> Result<(), BundleAbiError> {
        self.instance.commit()
    }

    pub fn commit_accepted(&mut self, records: &[BundleKeptRowsV1]) -> Result<(), BundleAbiError> {
        self.instance.commit_accepted(records)
    }

    pub fn cancel(&mut self) -> Result<(), BundleAbiError> {
        self.instance.cancel()
    }

    pub fn destroy(&mut self) -> Result<(), BundleAbiError> {
        self.instance.destroy()
    }

    /// Prevents finalizers from rewriting storage still borrowed by host work.
    pub fn poison_pending_execution(&mut self) {
        self.instance.poison_pending_execution();
    }

    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.instance.is_poisoned()
    }

    #[must_use]
    pub fn library_ref_count(&self) -> usize {
        Arc::strong_count(&self.library)
    }
}

impl Drop for LoadedBundleInstanceV1 {
    fn drop(&mut self) {
        self.instance.finalize_for_drop();
        if self.instance.is_poisoned() {
            // A terminal callback may have left native resources live, so keep its code resident.
            std::mem::forget(Arc::clone(&self.library));
        }
    }
}

// One private copy of a verified module, staged where no package writer
// reaches. The copy is the whole guarantee: the bytes come from the caller's
// authenticated vector rather than from a second lookup of the package path,
// and what holds them is out of reach of anything that can write that path.
// The two platforms spell "out of reach" differently, so each states its own
// terms below, and each states what its terms do not cover.

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn stage_error_v1(display: &str, error: &std::io::Error) -> BundleLoaderError {
    BundleLoaderError::StageImage {
        path: display.to_owned(),
        message: error.to_string(),
    }
}

/// The Linux image is an anonymous memory file, named through `/proc`.
///
/// It never had a directory entry, so there is no name to replace, and it is
/// sealed against writing, growing, shrinking, and further sealing, so there is
/// nothing left to rewrite either, by this process or any other. The path the
/// loader is given names this process's own descriptor.
#[cfg(target_os = "linux")]
struct PrivateModuleImageV1 {
    path: std::path::PathBuf,
    // The image exists for as long as this value does, because the descriptor
    // its path names is the only thing keeping it alive.
    _image: std::fs::File,
}

#[cfg(target_os = "linux")]
impl PrivateModuleImageV1 {
    fn stage(bytes: &[u8], display: &str) -> Result<Self, BundleLoaderError> {
        use std::fs::File;
        use std::io::Write;
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

        let name = c"chopinfer-bundle-image";
        // SAFETY-REVIEWED: memfd_create reads one NUL-terminated static name and returns a new descriptor.
        let raw = unsafe {
            libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING)
        };
        if raw < 0 {
            return Err(stage_error_v1(display, &std::io::Error::last_os_error()));
        }
        // SAFETY-REVIEWED: memfd_create returned a fresh descriptor this value now uniquely owns.
        let mut file = File::from(unsafe { OwnedFd::from_raw_fd(raw) });
        file.write_all(bytes)
            .map_err(|error| stage_error_v1(display, &error))?;
        // Sealing is what makes the copy immutable rather than merely private:
        // no further write, resize, or unsealing is accepted by anyone,
        // including this process.
        let seals =
            libc::F_SEAL_SEAL | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;
        // SAFETY-REVIEWED: fcntl adds seals to a descriptor this value owns and reads no pointer argument.
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, seals) } != 0 {
            return Err(stage_error_v1(display, &std::io::Error::last_os_error()));
        }
        let path = std::path::PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
        Ok(Self { path, _image: file })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    /// There is nothing to confirm: the seals answered it before the load.
    ///
    /// The descriptor stays owned by the library so its loader path cannot be reused.
    fn confirm_loaded(&self, _bytes: &[u8], _display: &str) -> Result<(), BundleLoaderError> {
        Ok(())
    }
}

/// The macOS image is a private named copy retained through library unload.
///
/// dyld will not open a `/dev/fd` path at all. It refuses the descriptor
/// directory with `EACCES` before it looks at what the descriptor holds, so the
/// Linux answer of naming an anonymous image has no macOS spelling, and a real
/// directory entry is the only thing that loads. The entry is therefore put out
/// of reach of the writer this loader defends against, which is whoever can
/// write the installed package.
///
/// The copy goes in a directory made for it alone: named with 128 bits from the
/// system CSPRNG, created rather than reused, at mode 0700. The copy inside it
/// is created exclusively at mode 0400, which is the only open that gets write
/// access to it, so reopening it for writing is refused without a chmod first.
/// The entry, directory, and descriptor remain owned by the library. They are
/// removed after dlclose, or retained with a poisoned library until process exit.
///
/// What cannot be made impossible is checked afterwards rather than assumed:
/// the entry handed to dyld must still name the file this process wrote, which
/// is what a replaced entry breaks, and that file must still hold the
/// authenticated bytes, which is what a rewrite breaks. Either check failing
/// refuses the load. Past that, macOS validates every page of a mapped image
/// against its code signature as it faults in, so rewriting a module already
/// loaded does not substitute code, it kills the process.
///
/// None of this covers another process running as this user. Such a process can
/// chmod the directory back and can debug this one, and no macOS control this
/// loader can reach stops it. What this design does answer is a package whose
/// path or inode changes after it verified.
#[cfg(target_os = "macos")]
struct PrivateModuleImageV1 {
    name: StagedNameV1,
    /// The descriptor the copy was created and written through, kept open so
    /// the bytes can be read back afterwards without naming the entry again.
    image: std::fs::File,
    /// The device and inode written, which the entry must still name after the
    /// load.
    identity: (u64, u64),
}

/// The macOS directory entry, removed when its owning private image drops.
/// A library closes before releasing the image, including failed loads.
#[cfg(target_os = "macos")]
struct StagedNameV1 {
    file: std::path::PathBuf,
    directory: std::path::PathBuf,
}

#[cfg(target_os = "macos")]
impl Drop for StagedNameV1 {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.file);
        let _ = std::fs::remove_dir(&self.directory);
    }
}

#[cfg(target_os = "macos")]
const IMAGE_DIRECTORY_PREFIX_V1: &str = "chopinfer-bundle-image-";

#[cfg(target_os = "macos")]
impl PrivateModuleImageV1 {
    fn stage(bytes: &[u8], display: &str) -> Result<Self, BundleLoaderError> {
        use std::fs::{DirBuilder, OpenOptions};
        use std::io::Write;
        use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};

        let directory = std::env::temp_dir().join(format!(
            "{IMAGE_DIRECTORY_PREFIX_V1}{:032x}",
            unguessable_image_name_v1()
        ));
        // Created rather than reused, so an entry already sitting at this name
        // is a refusal and not something to load out of.
        DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .map_err(|error| stage_error_v1(display, &error))?;
        let name = StagedNameV1 {
            file: directory.join("module.dylib"),
            directory,
        };
        let user = effective_user_v1();
        let holding = std::fs::symlink_metadata(&name.directory)
            .map_err(|error| stage_error_v1(display, &error))?;
        if !holding.is_dir() || holding.uid() != user || holding.mode() & 0o7777 != 0o700 {
            return Err(unprivate_directory_error_v1(display));
        }
        let mut image = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o400)
            .custom_flags(libc::O_CLOEXEC)
            .open(&name.file)
            .map_err(|error| stage_error_v1(display, &error))?;
        image
            .write_all(bytes)
            .map_err(|error| stage_error_v1(display, &error))?;
        let written = image
            .metadata()
            .map_err(|error| stage_error_v1(display, &error))?;
        if !written.is_file()
            || written.uid() != user
            || written.nlink() != 1
            || written.mode() & 0o7777 != 0o400
            || written.len() != bytes.len() as u64
        {
            return Err(unprivate_image_error_v1(display));
        }
        let staged = Self {
            name,
            image,
            identity: (written.dev(), written.ino()),
        };
        // The entry about to be handed to dyld has to be the file just written.
        if entry_identity_v1(staged.path(), display)? != staged.identity {
            return Err(replaced_image_error_v1(display));
        }
        Ok(staged)
    }

    fn path(&self) -> &Path {
        &self.name.file
    }

    /// Confirms that what dyld opened by name was this copy, unchanged.
    ///
    /// Both halves of what a name can do are checked once the load is over. An
    /// entry that no longer names the written file would mean dyld opened
    /// something else, and a file that no longer holds the authenticated bytes
    /// would mean it opened this one after a rewrite. The bytes are read back
    /// through the descriptor this value already holds, so confirming the copy
    /// does not itself look the entry up for its contents.
    fn confirm_loaded(&self, bytes: &[u8], display: &str) -> Result<(), BundleLoaderError> {
        use std::os::unix::fs::{FileExt, MetadataExt};

        if entry_identity_v1(self.path(), display)? != self.identity {
            return Err(replaced_image_error_v1(display));
        }
        let loaded = self
            .image
            .metadata()
            .map_err(|error| stage_error_v1(display, &error))?;
        if loaded.nlink() != 1 || loaded.len() != bytes.len() as u64 {
            return Err(rewritten_image_error_v1(display));
        }
        let mut mapped = vec![0_u8; bytes.len()];
        self.image
            .read_exact_at(&mut mapped, 0)
            .map_err(|error| stage_error_v1(display, &error))?;
        if mapped != bytes {
            return Err(rewritten_image_error_v1(display));
        }
        Ok(())
    }
}

/// 128 bits from the system CSPRNG, so the staged name is not derivable.
#[cfg(target_os = "macos")]
fn unguessable_image_name_v1() -> u128 {
    let mut bytes = [0_u8; 16];
    // SAFETY-REVIEWED: arc4random_buf fills exactly the owned buffer it is given and cannot fail.
    unsafe { libc::arc4random_buf(bytes.as_mut_ptr().cast::<c_void>(), bytes.len()) };
    u128::from_ne_bytes(bytes)
}

#[cfg(target_os = "macos")]
fn effective_user_v1() -> u32 {
    // SAFETY-REVIEWED: geteuid reads no argument, returns this process's own identity, and cannot fail.
    unsafe { libc::geteuid() }
}

/// Names what the entry itself is, following no symlink standing in for it.
#[cfg(target_os = "macos")]
fn entry_identity_v1(path: &Path, display: &str) -> Result<(u64, u64), BundleLoaderError> {
    use std::os::unix::fs::MetadataExt;

    let entry = std::fs::symlink_metadata(path).map_err(|error| stage_error_v1(display, &error))?;
    Ok((entry.dev(), entry.ino()))
}

#[cfg(target_os = "macos")]
fn unprivate_directory_error_v1(display: &str) -> BundleLoaderError {
    BundleLoaderError::StageImage {
        path: display.to_owned(),
        message: "the staged loader image directory is not a directory this user owns at mode 0700"
            .to_owned(),
    }
}

#[cfg(target_os = "macos")]
fn unprivate_image_error_v1(display: &str) -> BundleLoaderError {
    BundleLoaderError::StageImage {
        path: display.to_owned(),
        message: "the staged loader image is not a single-link file this user owns at mode 0400"
            .to_owned(),
    }
}

#[cfg(target_os = "macos")]
fn replaced_image_error_v1(display: &str) -> BundleLoaderError {
    BundleLoaderError::StageImage {
        path: display.to_owned(),
        message: "the staged loader image entry no longer names the file this process wrote"
            .to_owned(),
    }
}

#[cfg(target_os = "macos")]
fn rewritten_image_error_v1(display: &str) -> BundleLoaderError {
    BundleLoaderError::StageImage {
        path: display.to_owned(),
        message: "the staged loader image no longer holds the bytes that verified".to_owned(),
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn dl_error_message() -> String {
    // SAFETY-REVIEWED: dlerror returns a loader-owned NUL-terminated diagnostic or null.
    let error = unsafe { libc::dlerror() };
    if error.is_null() {
        "unknown dynamic loader error".to_owned()
    } else {
        // SAFETY-REVIEWED: a non-null dlerror result is valid until the next loader call on this thread.
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::mem::size_of;

    #[cfg(target_os = "macos")]
    use super::BundleLoaderError;
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    use super::PrivateModuleImageV1;
    use super::{ApiTableReaderV1, checked_api_from_reader};
    use crate::{BUNDLE_ABI_VERSION_V1, BundleAbiError, BundleApiHeaderV1, BundleRawApiV1};

    /// One staging at a time, so a test that looks for leftover entries does
    /// not see another test's image part-way through its own staging.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    static STAGE_LOCK_V1: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct ShortAdvertisedTableV1 {
        full_table_reads: Cell<u32>,
    }

    impl ApiTableReaderV1 for ShortAdvertisedTableV1 {
        fn read_header(&self) -> BundleApiHeaderV1 {
            BundleApiHeaderV1 {
                struct_size: size_of::<BundleApiHeaderV1>() as u32,
                abi_version: BUNDLE_ABI_VERSION_V1,
            }
        }

        fn read_supported_table(&self) -> BundleRawApiV1 {
            self.full_table_reads
                .set(self.full_table_reads.get().saturating_add(1));
            BundleRawApiV1::empty()
        }
    }

    #[test]
    fn short_advertised_table_stops_before_a_full_table_read() {
        let reader = ShortAdvertisedTableV1 {
            full_table_reads: Cell::new(0),
        };

        assert!(matches!(
            checked_api_from_reader(&reader),
            Err(BundleAbiError::StructSize {
                surface: "bundle API",
                ..
            })
        ));
        assert_eq!(reader.full_table_reads.get(), 0);
    }

    #[test]
    fn retirement_tail_is_read_only_when_its_entire_slot_is_advertised() {
        struct Reader {
            size: u32,
            tail_reads: Cell<u32>,
        }
        extern "C" fn finish(_: crate::BundleHandleV1) -> crate::BundleStatusV1 {
            crate::BundleStatusV1::Success
        }
        // SAFETY-REVIEWED: the test callback ignores both borrowed ABI inputs.
        unsafe extern "C" fn create(
            _: *const crate::BundleHostServicesV1,
            _: crate::BundleByteViewV1,
        ) -> crate::BundleCreateResultV1 {
            crate::BundleCreateResultV1::failure(crate::BundleStatusV1::BundleError)
        }
        // SAFETY-REVIEWED: the test callback ignores every borrowed ABI input.
        unsafe extern "C" fn execute(
            _: crate::BundleHandleV1,
            _: crate::BundleByteViewV1,
            _: crate::BundleExecuteResultSinkV1,
        ) -> crate::BundleStatusV1 {
            crate::BundleStatusV1::Success
        }
        impl ApiTableReaderV1 for Reader {
            fn read_header(&self) -> BundleApiHeaderV1 {
                BundleApiHeaderV1 {
                    struct_size: self.size,
                    abi_version: BUNDLE_ABI_VERSION_V1,
                }
            }
            fn read_supported_table(&self) -> BundleRawApiV1 {
                BundleRawApiV1 {
                    struct_size: self.size,
                    abi_version: BUNDLE_ABI_VERSION_V1,
                    create: Some(create),
                    execute: Some(execute),
                    commit: Some(finish),
                    cancel: Some(finish),
                    destroy: Some(finish),
                }
            }
            fn read_retirement_callback(&self) -> Option<crate::BundleRetireRequestFnV1> {
                self.tail_reads.set(self.tail_reads.get() + 1);
                None
            }
        }
        for size in size_of::<BundleRawApiV1>()..=super::REQUEST_RETIREMENT_END_V1 {
            let reader = Reader {
                size: size as u32,
                tail_reads: Cell::new(0),
            };
            assert!(
                !checked_api_from_reader(&reader)
                    .unwrap()
                    .supports_request_retirement()
            );
            assert_eq!(
                reader.tail_reads.get(),
                u32::from(size == super::REQUEST_RETIREMENT_END_V1)
            );
        }
    }

    /// The loader maps its own copy, so the package file stops mattering.
    ///
    /// Both attacks are run against one staged image: the directory entry is
    /// replaced, and the original inode is rewritten in place through a second
    /// handle. Neither reaches the bytes the loader would open.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn neither_a_replaced_entry_nor_a_rewritten_inode_changes_the_staged_image() {
        use std::fs::{self, File, OpenOptions};
        use std::io::{Seek, SeekFrom, Write};

        const VERIFIED: &[u8] = b"verified module bytes";

        let _staging = STAGE_LOCK_V1
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let directory = std::env::temp_dir().join(format!(
            "runtime-bundle-abi-loader-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&directory).expect("the test directory is creatable");
        let package = directory.join("module.bin");
        fs::write(&package, VERIFIED).expect("the test file is writable");

        // What a verifier authenticates is the byte vector it digested.
        let authenticated = fs::read(&package).expect("the test file reads");
        assert_eq!(authenticated, VERIFIED);
        // A writer that still holds the package inode open across staging is
        // exactly the case an open verified descriptor does not cover.
        let mut resident = OpenOptions::new()
            .write(true)
            .open(&package)
            .expect("the test file opens for writing");

        let image = PrivateModuleImageV1::stage(&authenticated, "module.bin")
            .expect("this platform stages a private loader image");

        // Rewrite the same inode the package path named when it verified.
        resident
            .seek(SeekFrom::Start(0))
            .expect("the resident handle rewinds");
        resident
            .write_all(b"rewritten in place!!!")
            .expect("the resident handle writes");
        resident.flush().expect("the resident handle flushes");
        drop(resident);
        // And replace the directory entry, which is the other half of the race.
        let replacement = directory.join("replacement.bin");
        fs::write(&replacement, b"replaced entry bytes!").expect("the test file is writable");
        fs::rename(&replacement, &package).expect("the test file is replaceable");
        assert_ne!(
            fs::read(&package).expect("the package path reads"),
            VERIFIED,
            "the package file must actually have changed"
        );

        // The copy is sealed or read only, so nothing rewrites it between
        // staging and the loader's own open of it either.
        assert!(
            OpenOptions::new()
                .write(true)
                .open(image.path())
                .and_then(|mut image| image.write_all(b"x"))
                .is_err(),
            "the staged image must refuse a write"
        );
        assert!(
            File::create(image.path()).is_err(),
            "the staged image must refuse truncation"
        );
        assert_eq!(
            fs::read(image.path()).expect("the staged image reads"),
            VERIFIED,
            "the loader must map the bytes that verified"
        );

        drop(image);
        let _ = fs::remove_dir_all(&directory);
    }

    /// The private image stays out of reach until its owner drops.
    ///
    /// On Linux the path names this process's own sealed memory file, which
    /// never had a directory entry at all. On macOS it names a private copy in
    /// a directory made for it: unguessably named, owned by this user at mode
    /// 0700, holding one read-only file with a single link. Either way nothing
    /// that can write the package can reach it, the path still reads exactly
    /// the bytes that verified, and dropping the image releases its name.
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn the_staged_image_is_out_of_reach_and_leaves_no_name_after_drop() {
        use std::fs::{self, File, OpenOptions};
        use std::io::Write;

        const VERIFIED: &[u8] = b"unreachable module bytes";

        let _staging = STAGE_LOCK_V1
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let image = PrivateModuleImageV1::stage(VERIFIED, "module.bin")
            .expect("this platform stages a private loader image");

        #[cfg(target_os = "linux")]
        assert!(
            image.path().starts_with("/proc/self/fd"),
            "the loader path must name this process's own descriptor: {}",
            image.path().display()
        );
        #[cfg(target_os = "macos")]
        {
            use std::os::unix::fs::MetadataExt;

            let directory = image
                .path()
                .parent()
                .expect("the staged copy sits in a directory");
            let holding = fs::symlink_metadata(directory).expect("the staging directory reads");
            assert!(
                holding.is_dir(),
                "the staging directory must be a directory"
            );
            assert_eq!(
                holding.mode() & 0o7777,
                0o700,
                "the staging directory must admit nobody else"
            );
            assert_eq!(
                holding.uid(),
                super::effective_user_v1(),
                "the staging directory must belong to this user"
            );
            let staged = fs::symlink_metadata(image.path()).expect("the staged copy reads");
            assert_eq!(
                staged.mode() & 0o7777,
                0o400,
                "the staged copy must be read only"
            );
            assert_eq!(staged.nlink(), 1, "the staged copy must have one name");

            // The name comes from the system CSPRNG rather than from anything a
            // package writer could derive, so two stagings never collide and
            // neither is predictable from this process.
            let name = directory
                .file_name()
                .and_then(|name| name.to_str())
                .expect("the staging directory has a UTF-8 name");
            let random = name
                .strip_prefix(super::IMAGE_DIRECTORY_PREFIX_V1)
                .expect("the staging directory carries the image prefix");
            assert_eq!(random.len(), 32, "the staged name must carry 128 bits");
            assert!(
                random.chars().all(|digit| digit.is_ascii_hexdigit()),
                "the staged name must be hexadecimal: {random}"
            );
            let second = PrivateModuleImageV1::stage(VERIFIED, "module.bin")
                .expect("this platform stages a second private loader image");
            assert_ne!(
                second.path().parent(),
                image.path().parent(),
                "two stagings must not share a name"
            );
            drop(second);
        }

        assert!(
            OpenOptions::new()
                .write(true)
                .open(image.path())
                .and_then(|mut image| image.write_all(b"x"))
                .is_err(),
            "the loader path must refuse a write"
        );
        assert!(
            File::create(image.path()).is_err(),
            "the loader path must refuse truncation"
        );
        assert_eq!(
            fs::read(image.path()).expect("the staged image reads"),
            VERIFIED,
            "the loader path must still read the bytes that verified"
        );

        drop(image);
        // Nothing this process staged is left under a name.
        #[cfg(target_os = "macos")]
        {
            let leftover = fs::read_dir(std::env::temp_dir())
                .expect("the temporary directory lists")
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with(super::IMAGE_DIRECTORY_PREFIX_V1))
                })
                .collect::<Vec<_>>();
            assert!(
                leftover.is_empty(),
                "a staged image must leave no directory entry: {leftover:?}"
            );
        }
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn confirmation_keeps_the_private_image_until_its_owner_drops() {
        let _staging = STAGE_LOCK_V1
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let bytes = b"private image lifetime";
        let image = PrivateModuleImageV1::stage(bytes, "lifetime").unwrap();
        let path = image.path().to_owned();
        image.confirm_loaded(bytes, "lifetime").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        drop(image);
        assert!(
            !path.exists(),
            "dropping the private image releases its path"
        );
        #[cfg(target_os = "macos")]
        assert!(
            !path.parent().unwrap().exists(),
            "dropping the image removes its private directory"
        );
    }

    /// The macOS copy answers for its name after the load, not only before it.
    ///
    /// Both changes below need a process running as this user, which is exactly
    /// the boundary this design does not claim to hold: such a process can
    /// chmod the private directory back and rewrite what is in it. What is
    /// claimed is that a copy reached that way is refused rather than returned,
    /// so a load can be told what happened instead of running it.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_copy_changed_around_the_load_is_refused_rather_than_confirmed() {
        use std::fs::{self, OpenOptions, Permissions};
        use std::io::Write;
        use std::os::unix::fs::PermissionsExt;

        const VERIFIED: &[u8] = b"confirmed module bytes..";
        const REWRITTEN: &[u8] = b"rewritten module bytes!!";

        let _staging = STAGE_LOCK_V1
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        // The file this process wrote, rewritten in place behind a chmod. The
        // length is unchanged, so it is the bytes themselves that answer.
        let image = PrivateModuleImageV1::stage(VERIFIED, "module.bin")
            .expect("this platform stages a private loader image");
        fs::set_permissions(image.path(), Permissions::from_mode(0o600))
            .expect("this user can chmod its own staged copy");
        OpenOptions::new()
            .write(true)
            .open(image.path())
            .expect("the chmodded copy opens for writing")
            .write_all(REWRITTEN)
            .expect("the chmodded copy is writable");
        let error = image
            .confirm_loaded(VERIFIED, "module.bin")
            .expect_err("a rewritten copy is refused");
        assert!(
            matches!(&error, BundleLoaderError::StageImage { message, .. }
                if message.contains("no longer holds the bytes that verified")),
            "unexpected refusal: {error}"
        );

        // And the entry, replaced by a different file holding the same bytes,
        // so it is the identity of what dyld opened that answers.
        let image = PrivateModuleImageV1::stage(VERIFIED, "module.bin")
            .expect("this platform stages a private loader image");
        let elsewhere = image.path().with_file_name("replacement.dylib");
        fs::write(&elsewhere, VERIFIED).expect("this user can write in its own staging directory");
        fs::rename(&elsewhere, image.path()).expect("the staged entry is replaceable");
        let error = image
            .confirm_loaded(VERIFIED, "module.bin")
            .expect_err("a replaced entry is refused");
        assert!(
            matches!(&error, BundleLoaderError::StageImage { message, .. }
                if message.contains("no longer names the file this process wrote")),
            "unexpected refusal: {error}"
        );
    }
}
