//! Split lagged mixer and a reversed-chain negative control.
use kernel_build::CubinSpec;
use kernels_sm120::sinkhorn_hc::{lagged, lagged_split};

pub const FAMILY: &str = "sinkhorn_hc_lagged_split";
pub const CUBIN: &str = "sinkhorn_hc_hcl4x5120_lagged_split_sm120";
pub const MUTANT: &str = "sinkhorn_hc_hcl4x5120_lagged_split_reverse_sm120";

pub fn entry(probe: bool) -> lagged_split::Entry {
    lagged_split::entry(lagged::LAB_PROFILE, lagged::LAB, probe)
}

pub fn mutant() -> lagged_split::Entry {
    let mut e = entry(true);
    e.reference.symbol = e.reference.symbol.replace("_v1", "_reverse_v1");
    e
}

pub fn cubins() -> Vec<CubinSpec> {
    let served = lagged_split::cubin_spec(CUBIN.into(), &[entry(false), entry(true)]).unwrap();
    let mut reversed = lagged_split::cubin_spec(MUTANT.into(), &[mutant()]).unwrap();
    reversed
        .defines
        .push(("CHOP_HC_LAGGED_REVERSE_CHAINS".into(), "1".into()));
    crate::declared(&[CUBIN, MUTANT], vec![served, reversed])
}
