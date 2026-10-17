//! The NVIDIA management library, loaded at run time so chopinfer-top
//! still starts on a host without the driver. NVML reads driver state and
//! hardware counters; it creates no CUDA context and submits no GPU work.
// SAFETY-REVIEWED: All native calls use checked symbols and the vendor ABI types of nvml.h.
#![allow(unsafe_code)]

use std::ffi::{CStr, c_char, c_int, c_uint, c_ulonglong, c_void};

use super::{Activity, Device, Reading, Reasons};

const LIBRARY: &CStr = c"libnvidia-ml.so.1";

const SUCCESS: c_int = 0;
const ERROR_INSUFFICIENT_SIZE: c_int = 7;
const ERROR_GPU_IS_LOST: c_int = 15;
/// `NVML_VALUE_NOT_AVAILABLE` as an unsigned long long.
const NOT_AVAILABLE: c_ulonglong = c_ulonglong::MAX;

const TEMPERATURE_GPU: c_uint = 0;
const THRESHOLD_SLOWDOWN: c_uint = 1;
const CLOCK_SM: c_uint = 1;
const CLOCK_MEMORY: c_uint = 2;
const PSTATE_UNKNOWN: c_uint = 32;
const ERROR_UNCORRECTED: c_uint = 1;
const VOLATILE_ECC: c_uint = 0;
const FEATURE_ENABLED: c_uint = 1;
const NAME_BUFFER: usize = 96;
const VERSION_BUFFER: usize = 80;

/// `NVML_STRUCT_VERSION(data, ver)`: the struct size with the version in
/// the top byte.
const fn struct_version<T>(version: u32) -> c_uint {
    std::mem::size_of::<T>() as c_uint | (version << 24)
}

#[repr(C)]
#[derive(Default)]
struct Memory {
    total: c_ulonglong,
    free: c_ulonglong,
    used: c_ulonglong,
}

#[repr(C)]
#[derive(Default)]
struct MemoryV2 {
    version: c_uint,
    total: c_ulonglong,
    reserved: c_ulonglong,
    free: c_ulonglong,
    used: c_ulonglong,
}

#[repr(C)]
#[derive(Default)]
struct Utilization {
    gpu: c_uint,
    memory: c_uint,
}

#[repr(C)]
#[derive(Default)]
struct TemperatureV1 {
    version: c_uint,
    sensor: c_uint,
    temperature: c_int,
}

#[repr(C)]
#[derive(Default)]
struct MarginTemperature {
    version: c_uint,
    margin: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ProcessInfo {
    pid: c_uint,
    used_gpu_memory: c_ulonglong,
    gpu_instance_id: c_uint,
    compute_instance_id: c_uint,
}

#[repr(C)]
#[derive(Default)]
struct GpmSupport {
    version: c_uint,
    supported: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GpmMetric {
    metric: c_uint,
    status: c_int,
    value: f64,
    short_name: *mut c_char,
    long_name: *mut c_char,
    unit: *mut c_char,
}

/// `NVML_GPM_METRIC_MAX` of the CUDA 13 header. The driver reads only
/// `count` entries, so a header with a smaller maximum fits as well.
const GPM_METRIC_MAX: usize = 333;
const GPM_METRICS_GET_VERSION: c_uint = 1;
const GPM_SUPPORT_VERSION: c_uint = 1;

#[repr(C)]
struct GpmMetricsGet {
    version: c_uint,
    count: c_uint,
    sample1: *mut c_void,
    sample2: *mut c_void,
    metrics: [GpmMetric; GPM_METRIC_MAX],
}

/// The GPM metrics read, in [`Activity`] order: graphics activity, SM
/// activity, SM occupancy, any tensor activity, DRAM bandwidth, PCIe rx and
/// tx.
const GPM_METRICS: [c_uint; 7] = [1, 2, 3, 5, 10, 21, 20];

struct Library(*mut c_void);

impl Library {
    fn open(name: &CStr) -> Result<Self, String> {
        // SAFETY-REVIEWED: The name is terminated and dlopen owns the returned handle.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(format!("{} could not be loaded", name.to_string_lossy()));
        }
        Ok(Self(handle))
    }

    fn symbol(&self, name: &CStr) -> Option<*mut c_void> {
        // SAFETY-REVIEWED: The library remains live and the name is terminated.
        let symbol = unsafe { libc::dlsym(self.0, name.as_ptr()) };
        (!symbol.is_null()).then_some(symbol)
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY-REVIEWED: This is the live handle owned by this wrapper.
        unsafe {
            libc::dlclose(self.0);
        }
    }
}

type Handle = *mut c_void;
type Init = extern "C" fn() -> c_int;
type Text = extern "C" fn(*mut c_char, c_uint) -> c_int;
type Count = extern "C" fn(*mut c_uint) -> c_int;
type ByIndex = extern "C" fn(c_uint, *mut Handle) -> c_int;
type DeviceText = extern "C" fn(Handle, *mut c_char, c_uint) -> c_int;
type DeviceU32 = extern "C" fn(Handle, *mut c_uint) -> c_int;
type DeviceU64 = extern "C" fn(Handle, *mut c_ulonglong) -> c_int;
type DeviceKindU32 = extern "C" fn(Handle, c_uint, *mut c_uint) -> c_int;
type DeviceStruct<T> = extern "C" fn(Handle, *mut T) -> c_int;
type EccMode = extern "C" fn(Handle, *mut c_uint, *mut c_uint) -> c_int;
type EccErrors = extern "C" fn(Handle, c_uint, c_uint, *mut c_ulonglong) -> c_int;
type Processes = extern "C" fn(Handle, *mut c_uint, *mut ProcessInfo) -> c_int;
type GpmAlloc = extern "C" fn(*mut *mut c_void) -> c_int;
type GpmFree = extern "C" fn(*mut c_void) -> c_int;
type GpmSampleGet = extern "C" fn(Handle, *mut c_void) -> c_int;
type GpmMetrics = extern "C" fn(*mut GpmMetricsGet) -> c_int;

/// Resolves `name` from `library` as a function of type `T`.
macro_rules! function {
    ($library:expr, $name:expr, $ty:ty) => {{
        $library.symbol($name).map(|pointer| {
            // SAFETY-REVIEWED: Each call site names the nvml.h signature of this symbol.
            unsafe { std::mem::transmute::<*mut c_void, $ty>(pointer) }
        })
    }};
}

/// The entry points used. Only the first four are required.
struct Api {
    init: Init,
    shutdown: Init,
    count: Count,
    by_index: ByIndex,
    driver: Option<Text>,
    name: Option<DeviceText>,
    temperature_v: Option<DeviceStruct<TemperatureV1>>,
    temperature: Option<DeviceKindU32>,
    margin: Option<DeviceStruct<MarginTemperature>>,
    threshold: Option<DeviceKindU32>,
    fan: Option<DeviceU32>,
    energy: Option<DeviceU64>,
    power: Option<DeviceU32>,
    power_limit: Option<DeviceU32>,
    clock: Option<DeviceKindU32>,
    max_clock: Option<DeviceKindU32>,
    pstate: Option<DeviceU32>,
    reasons: Option<DeviceU64>,
    memory_v2: Option<DeviceStruct<MemoryV2>>,
    memory: Option<DeviceStruct<Memory>>,
    utilization: Option<DeviceStruct<Utilization>>,
    link_generation: Option<DeviceU32>,
    link_width: Option<DeviceU32>,
    max_link_generation: Option<DeviceU32>,
    max_link_width: Option<DeviceU32>,
    ecc_mode: Option<EccMode>,
    ecc_errors: Option<EccErrors>,
    compute: Option<Processes>,
    graphics: Option<Processes>,
    gpm_support: Option<DeviceStruct<GpmSupport>>,
    gpm_alloc: Option<GpmAlloc>,
    gpm_free: Option<GpmFree>,
    gpm_sample: Option<GpmSampleGet>,
    gpm_metrics: Option<GpmMetrics>,
}

impl Api {
    fn resolve(library: &Library) -> Result<Self, String> {
        let required = |name: &CStr| {
            library
                .symbol(name)
                .ok_or_else(|| format!("libnvidia-ml.so.1 has no {}", name.to_string_lossy()))
        };
        // SAFETY-REVIEWED: Each required symbol is transmuted to its nvml.h signature.
        let (init, shutdown, count, by_index) = unsafe {
            (
                std::mem::transmute::<*mut c_void, Init>(required(c"nvmlInit_v2")?),
                std::mem::transmute::<*mut c_void, Init>(required(c"nvmlShutdown")?),
                std::mem::transmute::<*mut c_void, Count>(required(c"nvmlDeviceGetCount_v2")?),
                std::mem::transmute::<*mut c_void, ByIndex>(required(
                    c"nvmlDeviceGetHandleByIndex_v2",
                )?),
            )
        };
        Ok(Self {
            init,
            shutdown,
            count,
            by_index,
            driver: function!(library, c"nvmlSystemGetDriverVersion", Text),
            name: function!(library, c"nvmlDeviceGetName", DeviceText),
            temperature_v: function!(
                library,
                c"nvmlDeviceGetTemperatureV",
                DeviceStruct<TemperatureV1>
            ),
            temperature: function!(library, c"nvmlDeviceGetTemperature", DeviceKindU32),
            margin: function!(
                library,
                c"nvmlDeviceGetMarginTemperature",
                DeviceStruct<MarginTemperature>
            ),
            threshold: function!(library, c"nvmlDeviceGetTemperatureThreshold", DeviceKindU32),
            fan: function!(library, c"nvmlDeviceGetFanSpeed", DeviceU32),
            energy: function!(library, c"nvmlDeviceGetTotalEnergyConsumption", DeviceU64),
            power: function!(library, c"nvmlDeviceGetPowerUsage", DeviceU32),
            power_limit: function!(library, c"nvmlDeviceGetEnforcedPowerLimit", DeviceU32),
            clock: function!(library, c"nvmlDeviceGetClockInfo", DeviceKindU32),
            max_clock: function!(library, c"nvmlDeviceGetMaxClockInfo", DeviceKindU32),
            pstate: function!(library, c"nvmlDeviceGetPerformanceState", DeviceU32),
            reasons: function!(
                library,
                c"nvmlDeviceGetCurrentClocksEventReasons",
                DeviceU64
            )
            .or_else(|| {
                function!(
                    library,
                    c"nvmlDeviceGetCurrentClocksThrottleReasons",
                    DeviceU64
                )
            }),
            memory_v2: function!(
                library,
                c"nvmlDeviceGetMemoryInfo_v2",
                DeviceStruct<MemoryV2>
            ),
            memory: function!(library, c"nvmlDeviceGetMemoryInfo", DeviceStruct<Memory>),
            utilization: function!(
                library,
                c"nvmlDeviceGetUtilizationRates",
                DeviceStruct<Utilization>
            ),
            link_generation: function!(library, c"nvmlDeviceGetCurrPcieLinkGeneration", DeviceU32),
            link_width: function!(library, c"nvmlDeviceGetCurrPcieLinkWidth", DeviceU32),
            max_link_generation: function!(
                library,
                c"nvmlDeviceGetMaxPcieLinkGeneration",
                DeviceU32
            ),
            max_link_width: function!(library, c"nvmlDeviceGetMaxPcieLinkWidth", DeviceU32),
            ecc_mode: function!(library, c"nvmlDeviceGetEccMode", EccMode),
            ecc_errors: function!(library, c"nvmlDeviceGetTotalEccErrors", EccErrors),
            compute: function!(
                library,
                c"nvmlDeviceGetComputeRunningProcesses_v3",
                Processes
            ),
            graphics: function!(
                library,
                c"nvmlDeviceGetGraphicsRunningProcesses_v3",
                Processes
            ),
            gpm_support: function!(
                library,
                c"nvmlGpmQueryDeviceSupport",
                DeviceStruct<GpmSupport>
            ),
            gpm_alloc: function!(library, c"nvmlGpmSampleAlloc", GpmAlloc),
            gpm_free: function!(library, c"nvmlGpmSampleFree", GpmFree),
            gpm_sample: function!(library, c"nvmlGpmSampleGet", GpmSampleGet),
            gpm_metrics: function!(library, c"nvmlGpmMetricsGet", GpmMetrics),
        })
    }
}

/// The status of each query of one reading, so a lost GPU is noticed.
#[derive(Default)]
struct Statuses {
    lost: bool,
}

impl Statuses {
    fn check(&mut self, code: c_int) -> bool {
        if code == ERROR_GPU_IS_LOST {
            self.lost = true;
        }
        code == SUCCESS
    }
}

fn text(buffer: &[c_char]) -> String {
    let bytes = buffer
        .iter()
        .take_while(|byte| **byte != 0)
        .map(|byte| *byte as u8)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// NVML, initialized, with a handle for each GPU.
pub struct Nvml {
    api: Api,
    handles: Vec<Handle>,
    // Dropped after `Drop::drop` shuts NVML down.
    _library: Library,
}

impl Nvml {
    /// Loads and initializes NVML, or says why it cannot.
    pub fn open() -> Result<Self, String> {
        let library = Library::open(LIBRARY)?;
        let api = Api::resolve(&library)?;
        let code = (api.init)();
        if code != SUCCESS {
            return Err(format!("NVML did not initialize (status {code})"));
        }
        let mut count = 0;
        let code = (api.count)(&mut count);
        if code != SUCCESS {
            (api.shutdown)();
            return Err(format!("NVML did not count the GPUs (status {code})"));
        }
        let mut handles = Vec::new();
        for index in 0..count {
            let mut handle = std::ptr::null_mut();
            if (api.by_index)(index, &mut handle) == SUCCESS {
                handles.push(handle);
            } else {
                handles.push(std::ptr::null_mut());
            }
        }
        Ok(Self {
            api,
            handles,
            _library: library,
        })
    }

    pub fn driver(&self) -> Option<String> {
        let driver = self.api.driver?;
        let mut buffer = [0 as c_char; VERSION_BUFFER];
        (driver(buffer.as_mut_ptr(), VERSION_BUFFER as c_uint) == SUCCESS).then(|| text(&buffer))
    }

    pub fn count(&self) -> usize {
        self.handles.len()
    }

    fn u32(&self, call: Option<DeviceU32>, handle: Handle, statuses: &mut Statuses) -> Option<u32> {
        let call = call?;
        let mut value = 0;
        statuses.check(call(handle, &mut value)).then_some(value)
    }

    fn u64(&self, call: Option<DeviceU64>, handle: Handle, statuses: &mut Statuses) -> Option<u64> {
        let call = call?;
        let mut value = 0;
        statuses.check(call(handle, &mut value)).then_some(value)
    }

    fn kind_u32(
        &self,
        call: Option<DeviceKindU32>,
        handle: Handle,
        kind: c_uint,
        statuses: &mut Statuses,
    ) -> Option<u32> {
        let call = call?;
        let mut value = 0;
        statuses
            .check(call(handle, kind, &mut value))
            .then_some(value)
    }

    /// What does not change about GPU `index`.
    pub fn device(&self, index: usize) -> Device {
        let handle = self.handles[index];
        let mut statuses = Statuses::default();
        let name = self.api.name.and_then(|name| {
            let mut buffer = [0 as c_char; NAME_BUFFER];
            (name(handle, buffer.as_mut_ptr(), NAME_BUFFER as c_uint) == SUCCESS)
                .then(|| text(&buffer))
        });
        let generation = self.u32(self.api.max_link_generation, handle, &mut statuses);
        let width = self.u32(self.api.max_link_width, handle, &mut statuses);
        let gpm = self.api.gpm_support.is_some_and(|support| {
            let mut value = GpmSupport {
                version: GPM_SUPPORT_VERSION,
                supported: 0,
            };
            support(handle, &mut value) == SUCCESS && value.supported != 0
        }) && self.api.gpm_alloc.is_some()
            && self.api.gpm_sample.is_some()
            && self.api.gpm_metrics.is_some()
            && self.api.gpm_free.is_some();
        Device {
            index: index as u32,
            name: name.unwrap_or_else(|| "GPU".to_owned()),
            memory_total: self.memory(handle, &mut statuses).map(|(total, _)| total),
            sm_max_mhz: self.kind_u32(self.api.max_clock, handle, CLOCK_SM, &mut statuses),
            pcie_max: generation.zip(width),
            gpm,
        }
    }

    /// `(total, used)` bytes: version 2 leaves out what the driver
    /// reserves, as nvidia-smi does.
    fn memory(&self, handle: Handle, statuses: &mut Statuses) -> Option<(u64, u64)> {
        if let Some(memory_v2) = self.api.memory_v2 {
            let mut value = MemoryV2 {
                version: struct_version::<MemoryV2>(2),
                ..MemoryV2::default()
            };
            if statuses.check(memory_v2(handle, &mut value)) {
                return Some((value.total, value.used));
            }
        }
        let memory = self.api.memory?;
        let mut value = Memory::default();
        statuses
            .check(memory(handle, &mut value))
            .then_some((value.total, value.used))
    }

    fn temperature(&self, handle: Handle, statuses: &mut Statuses) -> Option<i32> {
        if let Some(temperature_v) = self.api.temperature_v {
            let mut value = TemperatureV1 {
                version: struct_version::<TemperatureV1>(1),
                sensor: TEMPERATURE_GPU,
                temperature: 0,
            };
            if statuses.check(temperature_v(handle, &mut value)) {
                return Some(value.temperature);
            }
        }
        self.kind_u32(self.api.temperature, handle, TEMPERATURE_GPU, statuses)
            .map(|value| value as i32)
    }

    /// Degrees under the nearest slowdown threshold.
    fn headroom(
        &self,
        handle: Handle,
        temperature: Option<i32>,
        statuses: &mut Statuses,
    ) -> Option<i32> {
        if let Some(margin) = self.api.margin {
            let mut value = MarginTemperature {
                version: struct_version::<MarginTemperature>(1),
                margin: 0,
            };
            if statuses.check(margin(handle, &mut value)) {
                return Some(value.margin);
            }
        }
        let slowdown = self.kind_u32(self.api.threshold, handle, THRESHOLD_SLOWDOWN, statuses)?;
        Some(slowdown as i32 - temperature?)
    }

    fn processes(
        &self,
        call: Option<Processes>,
        handle: Handle,
        statuses: &mut Statuses,
    ) -> Vec<(u32, Option<u64>)> {
        let Some(call) = call else {
            return Vec::new();
        };
        let mut infos = vec![ProcessInfo::default(); 32];
        for _ in 0..3 {
            let mut count = infos.len() as c_uint;
            let code = call(handle, &mut count, infos.as_mut_ptr());
            if code == ERROR_INSUFFICIENT_SIZE {
                infos.resize(count as usize + 8, ProcessInfo::default());
                continue;
            }
            if !statuses.check(code) {
                return Vec::new();
            }
            return infos[..(count as usize).min(infos.len())]
                .iter()
                .map(|info| {
                    let memory =
                        (info.used_gpu_memory != NOT_AVAILABLE).then_some(info.used_gpu_memory);
                    (info.pid, memory)
                })
                .collect();
        }
        Vec::new()
    }

    /// One reading of GPU `index` at `at` seconds, without GPM figures.
    /// `power` is the instant draw; the poller replaces it with the average
    /// over the poll from `energy`.
    pub fn read(&self, index: usize, at: f64) -> Reading {
        let handle = self.handles[index];
        let mut statuses = Statuses::default();
        if handle.is_null() {
            return Reading {
                at,
                lost: true,
                ..Reading::default()
            };
        }
        let temperature = self.temperature(handle, &mut statuses);
        let headroom = self.headroom(handle, temperature, &mut statuses);
        let memory = self.memory(handle, &mut statuses);
        let utilization = self.api.utilization.and_then(|utilization| {
            let mut value = Utilization::default();
            statuses.check(utilization(handle, &mut value)).then(|| {
                (
                    f64::from(value.gpu) / 100.0,
                    f64::from(value.memory) / 100.0,
                )
            })
        });
        let ecc = self.api.ecc_mode.and_then(|mode| {
            let (mut current, mut pending) = (0, 0);
            statuses
                .check(mode(handle, &mut current, &mut pending))
                .then_some(current == FEATURE_ENABLED)
        });
        let ecc_uncorrected = if ecc == Some(true) {
            self.api.ecc_errors.and_then(|errors| {
                let mut value = 0;
                statuses
                    .check(errors(handle, ERROR_UNCORRECTED, VOLATILE_ECC, &mut value))
                    .then_some(value)
            })
        } else {
            None
        };
        let compute = self.processes(self.api.compute, handle, &mut statuses);
        let graphics = self.processes(self.api.graphics, handle, &mut statuses);
        let link = self
            .u32(self.api.link_generation, handle, &mut statuses)
            .zip(self.u32(self.api.link_width, handle, &mut statuses));
        let reading = Reading {
            at,
            temperature,
            headroom,
            fan: self.u32(self.api.fan, handle, &mut statuses),
            power: self
                .u32(self.api.power, handle, &mut statuses)
                .map(|milliwatts| f64::from(milliwatts) / 1000.0),
            energy: self.u64(self.api.energy, handle, &mut statuses),
            power_limit: self
                .u32(self.api.power_limit, handle, &mut statuses)
                .map(|milliwatts| f64::from(milliwatts) / 1000.0),
            sm_mhz: self.kind_u32(self.api.clock, handle, CLOCK_SM, &mut statuses),
            memory_mhz: self.kind_u32(self.api.clock, handle, CLOCK_MEMORY, &mut statuses),
            pstate: self
                .u32(self.api.pstate, handle, &mut statuses)
                .filter(|pstate| *pstate != PSTATE_UNKNOWN),
            reasons: self
                .u64(self.api.reasons, handle, &mut statuses)
                .map(Reasons),
            memory_used: memory.map(|(_, used)| used),
            utilization,
            pcie: link,
            ecc,
            ecc_uncorrected,
            processes: super::merge(&compute, &graphics),
            activity: None,
            lost: false,
        };
        Reading {
            lost: statuses.lost,
            ..reading
        }
    }

    /// A GPM sample buffer, freed when dropped.
    pub fn sample(&self) -> Option<Sample<'_>> {
        let alloc = self.api.gpm_alloc?;
        let mut raw = std::ptr::null_mut();
        (alloc(&mut raw) == SUCCESS && !raw.is_null()).then_some(Sample { nvml: self, raw })
    }

    /// Takes GPM sample `sample` of GPU `index`.
    pub fn take(&self, index: usize, sample: &mut Sample<'_>) -> bool {
        self.api
            .gpm_sample
            .is_some_and(|take| take(self.handles[index], sample.raw) == SUCCESS)
    }

    /// The GPM figures between two samples of one GPU, `earlier` first.
    pub fn activity(&self, earlier: &Sample<'_>, later: &Sample<'_>) -> Option<Activity> {
        let metrics = self.api.gpm_metrics?;
        let blank = GpmMetric {
            metric: 0,
            status: 0,
            value: 0.0,
            short_name: std::ptr::null_mut(),
            long_name: std::ptr::null_mut(),
            unit: std::ptr::null_mut(),
        };
        let mut request = Box::new(GpmMetricsGet {
            version: GPM_METRICS_GET_VERSION,
            count: GPM_METRICS.len() as c_uint,
            sample1: earlier.raw,
            sample2: later.raw,
            metrics: [blank; GPM_METRIC_MAX],
        });
        for (slot, metric) in request.metrics.iter_mut().zip(GPM_METRICS) {
            slot.metric = metric;
        }
        if metrics(&mut *request) != SUCCESS {
            return None;
        }
        let value = |index: usize| {
            let metric = &request.metrics[index];
            (metric.status == SUCCESS && metric.value.is_finite()).then_some(metric.value)
        };
        let share = |index: usize| value(index).map(|percent| (percent / 100.0).clamp(0.0, 1.0));
        // PCIe figures are MiB a second.
        let rate = |index: usize| value(index).map(|mebibytes| mebibytes.max(0.0) * 1_048_576.0);
        Some(Activity {
            kernels: share(0),
            sm: share(1),
            occupancy: share(2),
            tensor: share(3),
            dram: share(4),
            pcie_rx: rate(5),
            pcie_tx: rate(6),
        })
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        (self.api.shutdown)();
    }
}

/// A GPM sample buffer, which cannot outlive NVML.
pub struct Sample<'a> {
    nvml: &'a Nvml,
    raw: *mut c_void,
}

impl Drop for Sample<'_> {
    fn drop(&mut self) {
        if let Some(free) = self.nvml.api.gpm_free {
            free(self.raw);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structs_match_nvml_h() {
        assert_eq!(std::mem::size_of::<ProcessInfo>(), 24);
        assert_eq!(std::mem::size_of::<GpmMetric>(), 40);
        assert_eq!(
            std::mem::size_of::<GpmMetricsGet>(),
            24 + 40 * GPM_METRIC_MAX
        );
        assert_eq!(struct_version::<MemoryV2>(2), 40 | 2 << 24);
        assert_eq!(struct_version::<MarginTemperature>(1), 8 | 1 << 24);
        assert_eq!(struct_version::<TemperatureV1>(1), 12 | 1 << 24);
    }

    #[test]
    fn a_missing_library_is_reported_not_fatal() {
        let error = Library::open(c"libchopinfer-top-absent.so.0")
            .err()
            .unwrap();
        assert!(error.contains("could not be loaded"), "{error}");
    }

    #[test]
    fn names_end_at_the_first_nul() {
        let mut buffer = [0 as c_char; 8];
        for (slot, byte) in buffer.iter_mut().zip(b"RTX\0junk") {
            *slot = *byte as c_char;
        }
        assert_eq!(text(&buffer), "RTX");
    }
}
