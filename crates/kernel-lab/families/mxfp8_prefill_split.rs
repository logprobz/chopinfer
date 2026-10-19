//! Split full prefill tiles and exact fold tails, including generic layouts.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::mxfp8_prefill_split as split;
pub const FAMILY: &str = "mxfp8_prefill_split";
pub const CUBIN: &str = "mxfp8_prefill_split_lab_sm120";
pub fn cubins() -> Vec<CubinSpec> {
    let entries = crate::mxfp8_prefill::entries();
    let mut spec = split::cubin_spec(CUBIN, &entries).unwrap();
    for entry in entries.iter().filter(|e| e.query.k == 65536) {
        spec.entries.push(split::control_spec(entry));
    }
    crate::declared(&[CUBIN], vec![spec])
}
