//! Staged native folds, tail cases and a deliberately persistent-chain control.
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{mxfp8_fold, mxfp8_prefill};
pub const FAMILY: &str = "mxfp8_prefill";
pub const CUBIN: &str = "mxfp8_prefill_sm120";
pub fn entries() -> Vec<mxfp8_fold::Entry> {
    mxfp8_fold::lab_entries()
        .into_iter()
        .map(|mut e| {
            e.query.m_capacity = 257;
            e.symbol = format!("{}_prefill_reference_v1", e.symbol);
            e
        })
        .collect()
}
pub fn cubins() -> Vec<CubinSpec> {
    let entries = entries();
    let mut spec = mxfp8_prefill::cubin_spec(CUBIN, &entries).unwrap();
    for e in &entries {
        spec.entries.push(e.spec());
        if e.query.k == 65536 {
            spec.entries.push(mxfp8_prefill::control_spec(e).unwrap());
        }
    }
    crate::declared(&[CUBIN], vec![spec])
}
