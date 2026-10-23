//! The staged `probe_sm120` cubin that `kernel_harness::time_ab` launches
//! (its flush, read and gate entries), for the packed fold timing tests.
use kernel_harness::{Cubins, LabResult, Session};

/// The variable naming the staged `probe_sm120.cubin`.
pub const PROBE: &str = "CHOPINFER_KERNEL_PROBE_CUBIN";

/// `kernel_harness::gpu_test` on a session that also loads the timing probe
/// from [`PROBE`], whose digest must equal kernel-lab's locked
/// `probe_sm120`. Without a selected GPU it skips as `gpu_test` does.
pub fn timing_test<T>(
    need_bytes: u64,
    run: impl FnOnce(&Session) -> LabResult<T>,
) -> LabResult<Option<T>> {
    if kernel_harness::selected_ordinals()?.is_none() {
        return kernel_harness::gpu_test(need_bytes, run);
    }
    let path = std::env::var_os(PROBE)
        .ok_or_else(|| format!("{PROBE} must name the staged probe_sm120 cubin"))?;
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let lock = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels.lock");
    let records: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(lock).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let locked = records
        .iter()
        .find(|record| record["name"] == "probe_sm120")
        .and_then(|record| record["cubin_sha256"].as_str())
        .ok_or("kernel-lab's lock has no probe_sm120 digest")?;
    if kernel_build::digest(&bytes) != locked {
        return Err("the staged probe_sm120 differs from kernel-lab's lock".into());
    }
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let cubins = Cubins::new(
        Box::leak(vec![("probe_sm120", bytes)].into_boxed_slice()),
        &[],
    );
    kernel_harness::gpu_test_with(cubins, need_bytes, run)
}
