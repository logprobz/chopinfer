//! Captured-operand qualification: SIMT, block MMA, native FP4 and f64.
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernels_sm120::pooled_indexer::{position, position_live};
pub const FAMILY: &str = "position_mma_qual";
pub const CUBIN: &str = "position_mma_qual_sm120";
pub fn entry(ratio: u32) -> position::Entry {
    let mut e = super::position_mma_probe::entry(ratio);
    e.name = format!("mma_qual_reference_r{ratio}");
    e
}
pub fn block(ratio: u32) -> String {
    format!("chop_position_mma_qual_bf16_r{ratio}_v1")
}
pub fn native(ratio: u32) -> String {
    format!("chop_position_mma_native_r{ratio}_v1")
}
pub fn oracle(ratio: u32) -> String {
    format!("chop_position_f64_probe_r{ratio}_v1")
}
pub const PACK: &str = "chop_position_mma_pack_query_v1";
fn ptr(name: &str, ty: &str) -> P {
    P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}
fn u32p(name: &str) -> P {
    P::U32 { name: name.into() }
}
fn u64p(name: &str) -> P {
    P::U64 { name: name.into() }
}
pub fn cubins() -> Vec<CubinSpec> {
    let mut spec = super::position_mma_probe::cubins().remove(0);
    spec.name = CUBIN.into();
    for entry in &mut spec.entries {
        entry.symbol = entry
            .symbol
            .replace("mma_reference", "mma_qual_reference")
            .replace("chop_position_mma_probe_", "chop_position_mma_qual_bf16_");
    }
    spec.headers.extend([
        "#include \"pooled_indexer/position_mma_native.cuh\"".into(),
        "#include \"pooled_indexer/position_f64_probe.cuh\"".into(),
    ]);
    spec.entries.push(EntrySpec::new(
        PACK.into(),
        vec![
            ptr("q", "const uint16_t"),
            ptr("live", "const uint32_t"),
            ptr("codes", "uint8_t"),
            ptr("scales", "uint8_t"),
            ptr("status", "uint32_t"),
        ],
        "chop::position_mma_native::pack(q,live,codes,scales,status);".into(),
        256,
        1,
    ));
    for ratio in [1, 2] {
        let mut candidate = position_live::dense_specs(&entry(ratio), false)
            .unwrap()
            .remove(0);
        candidate.symbol = native(ratio);
        candidate.params.extend([
            ptr("qcodes", "const uint8_t"),
            ptr("qscales", "const uint8_t"),
            ptr("qstatus", "const uint32_t"),
        ]);
        candidate.body = candidate
            .body
            .replace(
                "chop::position_live::dense_scores",
                "chop::position_mma_native::scores",
            )
            .replace(");", ",qcodes,qscales,qstatus);");
        spec.entries.push(candidate);
        spec.entries.push(EntrySpec::new(oracle(ratio),vec![ptr("q","const uint16_t"),ptr("weights","const float"),ptr("codes","const uint8_t"),ptr("scales","const uint8_t"),ptr("tables","const int32_t"),ptr("positions","const int32_t"),ptr("live","const uint32_t"),ptr("output","double"),ptr("magnitudes","double"),u32p("width"),u64p("code_stride"),u64p("scale_stride")],format!("chop::position_f64_probe::scores<{},{}>(q,weights,codes,scales,tables,positions,live,output,magnitudes,width,code_stride,scale_stride);",64/ratio,ratio),128,1));
    }
    crate::declared(&[CUBIN], vec![spec])
}
