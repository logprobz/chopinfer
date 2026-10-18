//! Shared MXFP8 row decode and a truncating device negative control.

use kernel_build::CubinSpec;
use kernels_sm120::row::mxfp8_decode;

pub const FAMILY: &str = "mxfp8_rows";
pub const MUTANT_CUBIN: &str = "row_mxfp8_decode_truncate_sm120";
pub const MUTANT: &str = "chop_row_mxfp8_decode_truncate_v1";

pub fn cubins() -> Vec<CubinSpec> {
    let mut mutant = mxfp8_decode::cubin_spec();
    mutant.name = MUTANT_CUBIN.into();
    mutant.entries = vec![mxfp8_decode::entry_spec(MUTANT)];
    mutant.entries[0].body =
        "chop::row::mxfp8_decode::rows<true>(codes,scales,live_rows,output,width);".into();
    crate::declared(
        &[mxfp8_decode::CUBIN, MUTANT_CUBIN],
        vec![mxfp8_decode::cubin_spec(), mutant],
    )
}
