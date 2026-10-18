//! The clamped SwiGLU with BF16 rounding and ceil E4M3/32 quantization's
//! lab cubin (`row::silu_mul_quantize::lab_cubin_specs`), and the probe of
//! its exponential's error (`cuda/probes/expf_error.cuh`), built with the
//! same flags.

use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernels_sm120::row::silu_mul_quantize::LAB_CUBIN;
use std::path::PathBuf;

pub const FAMILY: &str = "silu_mul_quantize";
pub const EXPF_CUBIN: &str = "probe_expf_error_sm120";
pub const EXPF: &str = "chop_probe_expf_error_v1";
/// Threads of an expf error CTA.
pub const EXPF_THREADS: u32 = 256;
/// The distance buckets per class of the probe's results (`DISTANCES`).
pub const EXPF_DISTANCES: usize = 8;

pub fn cubins() -> Vec<CubinSpec> {
    let pointer = |name: &str, cxx_type: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let u64 = |name: &str| P::U64 { name: name.into() };
    let u32 = |name: &str| P::U32 { name: name.into() };
    let probe = CubinSpec {
        name: EXPF_CUBIN.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."),
        include_dirs: vec![],
        headers: vec!["#include \"kernel-lab/cuda/probes/expf_error.cuh\"".into()],
        defines: vec![],
        entries: vec![EntrySpec::new(
            EXPF.into(),
            vec![
                u32("start"),
                u64("count"),
                u32("clamp_bits"),
                u32("flag_steps"),
                pointer("results", "unsigned long long"),
                pointer("flagged", "unsigned long long"),
                u32("flag_capacity"),
                pointer("values", "float"),
            ],
            "chop::probe::expf_error::measure(start, count, clamp_bits, flag_steps, results, flagged, flag_capacity, values);".into(),
            EXPF_THREADS,
            1,
        )],
    };
    let mut specs = kernels_sm120::row::silu_mul_quantize::lab_cubin_specs();
    specs.push(probe);
    crate::declared(&[LAB_CUBIN, EXPF_CUBIN], specs)
}
