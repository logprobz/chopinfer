//! PDL producer probes for consumers of the model-neutral kernel harness.
//! Test crates build these separately from their serving cubins.

use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};

pub const CUBIN: &str = "dependency_probes_sm120";
pub const COPY: &str = "chop_dependency_probe_copy_v1";
pub const LIVE: &str = "chop_dependency_probe_live_v1";

pub fn cubin_spec() -> CubinSpec {
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let u64p = |name: &str| P::U64 { name: name.into() };
    let u32p = |name: &str| P::U32 { name: name.into() };
    CubinSpec {
        name: CUBIN.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"probes/dependencies.cuh\"".into()],
        defines: vec![],
        entries: vec![
            EntrySpec::new(COPY.into(), vec![ptr("src", "const uint32_t"), ptr("dst", "uint32_t"), u64p("words"), u64p("spin")],
                "chop::probes::copy(src,dst,words,spin);".into(), 256, 1),
            EntrySpec::new(LIVE.into(), vec![ptr("eviction", "const uint4"), ptr("checksum", "uint32_t"), u64p("vectors"), ptr("operands", "const chop::probes::Operand"), u32p("operand_count"), ptr("live", "uint32_t"), u32p("rows"), u64p("tail_ns")],
                "chop::probes::live(eviction,checksum,vectors,operands,operand_count,live,rows,tail_ns);".into(), 128, 1),
        ],
    }
}
