//! NVML telemetry outside the measured event interval.
// SAFETY-REVIEWED: All native calls use checked symbols and matching vendor ABI types.
#![allow(unsafe_code)]

use crate::LabResult;
use serde_json::{Value, json};
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

struct Library(*mut c_void);
impl Library {
    fn open(name: &CStr) -> LabResult<Self> {
        // SAFETY-REVIEWED: The name is terminated and dlopen owns the returned handle.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(format!("cannot load {}", name.to_string_lossy()));
        }
        Ok(Self(handle))
    }
    fn symbol(&self, name: &CStr) -> LabResult<*mut c_void> {
        // SAFETY-REVIEWED: The library remains live and the name is terminated.
        let symbol = unsafe { libc::dlsym(self.0, name.as_ptr()) };
        if symbol.is_null() {
            return Err(format!("missing {}", name.to_string_lossy()));
        }
        Ok(symbol)
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
macro_rules! function {
    ($lib:expr, $name:expr, $ty:ty) => {{
        let pointer = $lib.symbol($name)?;
        // SAFETY-REVIEWED: Each call site specifies the documented ABI for this symbol.
        unsafe { std::mem::transmute::<*mut c_void, $ty>(pointer) }
    }};
}
fn check(code: c_int) -> LabResult<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("GPU telemetry status {code}"))
    }
}

pub fn l2_bytes(ordinal: i32) -> LabResult<usize> {
    let lib = Library::open(c"libcuda.so.1")?;
    let get = function!(
        lib,
        c"cuDeviceGet",
        extern "C" fn(*mut c_int, c_int) -> c_int
    );
    let attr = function!(
        lib,
        c"cuDeviceGetAttribute",
        extern "C" fn(*mut c_int, c_int, c_int) -> c_int
    );
    let (mut device, mut bytes) = (0, 0);
    check(get(&mut device, ordinal))?;
    check(attr(&mut bytes, 38, device))?;
    usize::try_from(bytes).map_err(|e| e.to_string())
}

pub struct Monitor {
    lib: Library,
    device: *mut c_void,
}
impl Monitor {
    pub fn open(uuid: &str) -> LabResult<Self> {
        let lib = Library::open(c"libnvidia-ml.so.1")?;
        let init = function!(lib, c"nvmlInit_v2", extern "C" fn() -> c_int);
        let shutdown = function!(lib, c"nvmlShutdown", extern "C" fn() -> c_int);
        let get = function!(
            lib,
            c"nvmlDeviceGetHandleByUUID",
            extern "C" fn(*const c_char, *mut *mut c_void) -> c_int
        );
        let uuid = CString::new(uuid).map_err(|e| e.to_string())?;
        let mut device = std::ptr::null_mut();
        check(init())?;
        if let Err(error) = check(get(uuid.as_ptr(), &mut device)) {
            shutdown();
            return Err(error);
        }
        Ok(Self { lib, device })
    }
    pub fn snapshot(&self, session_sm: u64) -> Value {
        let query = |name: &CStr| -> LabResult<u32> {
            let get = function!(
                self.lib,
                name,
                extern "C" fn(*mut c_void, *mut c_uint) -> c_int
            );
            let mut value = 0;
            check(get(self.device, &mut value))?;
            Ok(value)
        };
        let clock = |kind| -> LabResult<u32> {
            let get = function!(
                self.lib,
                c"nvmlDeviceGetClockInfo",
                extern "C" fn(*mut c_void, c_uint, *mut c_uint) -> c_int
            );
            let mut value = 0;
            check(get(self.device, kind, &mut value))?;
            Ok(value)
        };
        let reasons = (|| -> LabResult<u64> {
            let get = function!(
                self.lib,
                c"nvmlDeviceGetCurrentClocksThrottleReasons",
                extern "C" fn(*mut c_void, *mut u64) -> c_int
            );
            let mut value = 0;
            check(get(self.device, &mut value))?;
            Ok(value)
        })();
        let temperature = (|| -> LabResult<u32> {
            let get = function!(
                self.lib,
                c"nvmlDeviceGetTemperature",
                extern "C" fn(*mut c_void, c_uint, *mut c_uint) -> c_int
            );
            let mut value = 0;
            check(get(self.device, 0, &mut value))?;
            Ok(value)
        })();
        let sm = clock(1);
        let mut flags = round_flags(
            sm.as_ref().ok().copied(),
            reasons.as_ref().ok().copied(),
            session_sm,
        );
        let memory = clock(2);
        let pstate = query(c"nvmlDeviceGetPerformanceState");
        let power = query(c"nvmlDeviceGetPowerUsage");
        let limit = query(c"nvmlDeviceGetPowerManagementLimit");
        if (memory.is_err()
            || pstate.is_err()
            || power.is_err()
            || limit.is_err()
            || temperature.is_err())
            && !flags.contains(&"telemetry_unavailable")
        {
            flags.push("telemetry_unavailable");
        }
        json!({"sm_mhz": sm, "memory_mhz": memory, "clock_event_reasons": reasons,
            "pstate": pstate, "power_mw": power, "power_limit_mw": limit,
            "temperature_c": temperature, "flags": flags})
    }
}
impl Drop for Monitor {
    fn drop(&mut self) {
        if let Ok(pointer) = self.lib.symbol(c"nvmlShutdown") {
            // SAFETY-REVIEWED: The checked symbol has NVML's documented shutdown ABI.
            let shutdown =
                unsafe { std::mem::transmute::<*mut c_void, extern "C" fn() -> c_int>(pointer) };
            shutdown();
        }
    }
}

fn round_flags(sm: Option<u32>, reasons: Option<u64>, baseline: u64) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if reasons.is_some_and(|mask| mask & (0x4 | 0x8 | 0x20 | 0x40 | 0x80) != 0) {
        flags.push("power_or_thermal_throttle");
    }
    if sm.is_some_and(|clock| {
        baseline > 0 && u64::from(clock).abs_diff(baseline) * 100 > baseline * 5
    }) {
        flags.push("sm_clock_drift");
    }
    if sm.is_none() || reasons.is_none() {
        flags.push("telemetry_unavailable");
    }
    flags
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn throttle_and_clock_thresholds() {
        assert!(round_flags(Some(950), Some(0x1), 1000).is_empty());
        assert_eq!(
            round_flags(Some(949), Some(0x4), 1000),
            vec!["power_or_thermal_throttle", "sm_clock_drift"]
        );
        assert_eq!(round_flags(None, None, 1000), vec!["telemetry_unavailable"]);
        for mask in [0x8, 0x20, 0x40, 0x80] {
            assert_eq!(
                round_flags(Some(1000), Some(mask), 1000),
                vec!["power_or_thermal_throttle"]
            );
        }
    }
}
