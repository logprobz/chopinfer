//! Prefill row tiles and a changed-butterfly negative control.
use kernel_build::CubinSpec;
use kernels_sm120::sinkhorn_hc::{lagged, lagged_tiled, tiled};

pub const FAMILY: &str = "sinkhorn_hc_lagged_tiled";
pub const CUBIN: &str = "sinkhorn_hc_hcl4x5120_lagged_tiled_sm120";
pub const MUTANT: &str = "sinkhorn_hc_hcl4x5120_lagged_tiled_reverse_sm120";

pub fn entry(rows: u32, probe: bool) -> lagged_tiled::Entry {
    lagged_tiled::entry(lagged::LAB_PROFILE, lagged::LAB, rows, 256, probe)
}

pub fn mutant() -> lagged_tiled::Entry {
    let mut e = entry(8, true);
    e.reference.symbol = e.reference.symbol.replace("_v1", "_reverse_v1");
    e
}

pub fn cubins() -> Vec<CubinSpec> {
    let entries: Vec<_> = tiled::ROWS.into_iter()
        .flat_map(|rows| [entry(rows, false), entry(rows, true)]).collect();
    let served = lagged_tiled::cubin_spec(CUBIN.into(), &entries).unwrap();
    let mut reversed = lagged_tiled::cubin_spec(MUTANT.into(), &[mutant()]).unwrap();
    reversed.defines.push((tiled::LEVELS_REVERSED_DEFINE.into(), "1".into()));
    crate::declared(&[CUBIN, MUTANT], vec![served, reversed])
}
