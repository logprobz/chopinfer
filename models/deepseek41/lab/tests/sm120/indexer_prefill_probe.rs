//! Small device comparisons keep capacity-sized score tails off the host.
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};

fn ptr(name: &str, ty: &str) -> P {
    P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}
fn u32_arg(name: &str) -> P {
    P::U32 { name: name.into() }
}

pub fn spec() -> CubinSpec {
    let fill = EntrySpec::new("chop_index_prefill_fill_v1".into(),
        vec![ptr("out", "uint32_t"), P::U64 { name: "words".into() }, u32_arg("value")],
        "for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words; i += uint64_t(gridDim.x) * blockDim.x) out[i] = value;".into(), 256, 1);
    let compare = EntrySpec::new("chop_index_prefill_compare_v1".into(),
        vec![ptr("old", "const uint32_t"), ptr("now", "const uint32_t"), ptr("counts", "uint32_t"),
             u32_arg("rows"), u32_arg("live"), u32_arg("old_width"), u32_arg("width"), u32_arg("first"), u32_arg("counter")],
        r#"uint32_t bad = 0, tail = 0;
        const uint32_t row = blockIdx.y;
        if (row >= rows) return;
        for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < old_width; i += gridDim.x * blockDim.x) {
          const uint32_t a = old[uint64_t(row) * old_width + i];
          if (i < width) bad += a != now[uint64_t(row + first) * width + i];
          else tail += a != (row < live ? 0xff800000u : 0xa5a5a5a5u);
        }
        if (bad) atomicOr(counts + counter, 1u);
        if (tail) atomicOr(counts + 7, 1u);"#.into(), 256, 1);
    CubinSpec {
        name: "indexer_prefill_probe".into(),
        sources_root: kernels_sm120::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include <stdint.h>".into()],
        defines: vec![],
        entries: vec![fill, compare],
    }
}
