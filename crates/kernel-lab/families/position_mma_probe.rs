//! Lab-only tensor-core arithmetic experiment. No serving registration.
use kernel_build::CubinSpec;
use kernels_sm120::pooled_indexer::{position,position_live};
pub const FAMILY: &str = "position_mma_probe";
pub const CUBIN: &str = "position_mma_probe_sm120";
pub fn entry(ratio: u32) -> position::Entry {
    let mut e = super::position_live::dense(ratio);
    e.name = format!("mma_reference_r{ratio}");
    e
}
pub fn symbol(ratio:u32) -> String { format!("chop_position_mma_probe_r{ratio}_v1") }
pub fn cubins() -> Vec<CubinSpec> {
    let mut entries = vec![];
    for ratio in [1,2] {
        let e = entry(ratio);
        let baseline = position_live::dense_specs(&e,false).unwrap();
        let mut candidate = baseline[0].clone();
        candidate.symbol = symbol(ratio);
        candidate.body = candidate.body.replace("chop::position_live::dense_scores", "chop::position_mma_probe::scores");
        entries.extend(baseline);
        entries.push(candidate);
    }
    let mut spec = position_live::cubin_spec(CUBIN,entries);
    spec.headers.push("#include \"pooled_indexer/position_mma_probe.cuh\"".into());
    crate::declared(&[CUBIN],vec![spec])
}
