//! PDL producers owned by the compressor lab, never loaded by serving.

use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use std::path::Path;

pub const FAMILY: &str = "compressor_probes";
pub const CUBIN: &str = "deepseek41_compressor_probes_v1";
pub const LIVE: &str = "chop_deepseek41_compressor_probe_live_v1";
pub const COPY: &str = "chop_deepseek41_compressor_probe_copy_v1";

pub fn cubin(lab: &Path) -> CubinSpec {
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let u32 = |name: &str| P::U32 { name: name.into() };
    let u64 = |name: &str| P::U64 { name: name.into() };
    CubinSpec {
        name: CUBIN.into(),
        sources_root: lab.into(),
        include_dirs: vec!["cuda".into()],
        headers: vec!["#include \"compressor_probes.cuh\"".into()],
        defines: vec![],
        entries: vec![
            EntrySpec::new(
                LIVE.into(),
                vec![ptr("eviction", "const uint4"), ptr("checksum", "uint32_t"),
                    u64("vectors"), ptr("operands", "const chop::deepseek41::compressor_probe::Operand"),
                    u32("operand_count"), ptr("count", "uint32_t"), u32("rows"), u64("tail_ns")],
                "chop::deepseek41::compressor_probe::live(eviction,checksum,vectors,operands,operand_count,count,rows,tail_ns);".into(),
                128, 1,
            ),
            EntrySpec::new(
                COPY.into(),
                vec![ptr("source", "const uint32_t"), ptr("target", "uint32_t"), u64("words"), u64("cycles")],
                "chop::deepseek41::compressor_probe::copy(source,target,words,cycles);".into(),
                256, 1,
            ),
        ],
    }
}
