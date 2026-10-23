//! Split prefill full tiles and fold tails reading
//! `mxfp8-fold-packed-8x128-v1` weight codes, beside the row-major entries
//! they must equal bit for bit. The cubin also holds the unsplit tiles, which
//! the split op launches for rows below one tile or in whole tiles.

use crate::mxfp8_prefill_packed::{SHAPES, TIMING_SHAPES, entry};
use kernel_build::CubinSpec;
use kernels_sm120::gemm::{mxfp8_fold::Entry, mxfp8_prefill, mxfp8_prefill_split as split};

pub const FAMILY: &str = "mxfp8_prefill_split_packed";
pub const CUBIN: &str = "chop_mxfp8_prefill_split_packed_lab";

pub fn entries() -> Vec<Entry> {
    SHAPES.into_iter().map(|s| entry(FAMILY, s)).collect()
}

pub fn timing_entries() -> Vec<Entry> {
    TIMING_SHAPES
        .into_iter()
        .map(|s| entry(FAMILY, s))
        .collect()
}

pub fn cubins() -> Vec<CubinSpec> {
    let entries = [entries(), timing_entries()].concat();
    let mut spec = split::cubin_spec(CUBIN, &entries).unwrap();
    for e in &entries {
        spec.entries.extend(mxfp8_prefill::specs(e).unwrap());
        spec.entries.push(mxfp8_prefill::packed_spec(e).unwrap());
        spec.entries.extend(split::packed_specs(e).unwrap());
    }
    crate::declared(&[CUBIN], vec![spec])
}
