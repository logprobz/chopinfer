//! The SM120 payload admission (`bundle/cuda/admission/payload.cuh`): one
//! entry per check of `crate::payload`, its integer rules the chained
//! engine's and the shared tiled GEMM's admission rules
//! (`kernels_sm120::moe::chained::admit_scales`,
//! `chained::formats::admit_ue4m3_scales` and `admit_mxfp8_payload`).

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};

use crate::payload::Check;

/// The lock family of [`CUBIN`] (`bundle/kernels.lock.d`).
pub const FAMILY: &str = "payload_admission";
/// The admission cubin.
pub const CUBIN: &str = "payload_admission_glm5next_sm120";
/// Threads of an admission CTA.
pub const THREADS: u32 = 256;
/// Most CTAs of one admission launch; each thread strides over the rest.
pub const MAX_BLOCKS: u32 = 4096;

/// `check`'s entry of [`CUBIN`].
pub fn symbol(check: Check) -> &'static str {
    match check {
        Check::F32Scales => "chop_admit_f32_scales_glm5next_v1",
        Check::Ue4m3Scales => "chop_admit_ue4m3_scales_glm5next_v1",
        Check::Mxfp8 => "chop_admit_mxfp8_payload_glm5next_v1",
    }
}

/// `check`'s entry definition.
pub fn entry(check: Check) -> EntrySpec {
    let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let count = EntryParam::U64 {
        name: "count".into(),
    };
    let first = ptr("first", "uint32_t");
    let (params, body) = match check {
        Check::F32Scales => (
            vec![ptr("words", "const uint32_t"), count, first],
            "chop::glm5next::admission::f32_scales(words, count, first);",
        ),
        Check::Ue4m3Scales => (
            vec![ptr("codes", "const uint8_t"), count, first],
            "chop::glm5next::admission::ue4m3_scales(codes, count, first);",
        ),
        Check::Mxfp8 => (
            vec![
                ptr("codes", "const uint8_t"),
                ptr("scales", "const uint8_t"),
                count,
                first,
            ],
            "chop::glm5next::admission::mxfp8_payload(codes, scales, count, first);",
        ),
    };
    EntrySpec::new(symbol(check).into(), params, body.into(), THREADS, 1)
}

/// The launch of one admission over all `count` elements.
pub fn plan(count: u64) -> Result<LaunchPlan, Refusal> {
    let blocks = count
        .div_ceil(u64::from(THREADS))
        .clamp(1, u64::from(MAX_BLOCKS));
    let plan = LaunchPlan {
        grid: [blocks as u32, 1, 1],
        block: [THREADS, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

/// [`CUBIN`]: every check's entry.
pub fn cubin_spec() -> CubinSpec {
    CubinSpec {
        name: CUBIN.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        include_dirs: vec!["models/glm5next/bundle/cuda".into()],
        headers: vec!["#include \"admission/payload.cuh\"".into()],
        defines: vec![],
        entries: Check::ALL.map(entry).to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_check_has_an_entry_and_large_views_stride() {
        assert_eq!(cubin_spec().entries.len(), 3);
        assert_eq!(plan(1 << 30).unwrap().grid, [MAX_BLOCKS, 1, 1]);
        assert_eq!(plan(0).unwrap().grid, [1, 1, 1]);
    }
}
