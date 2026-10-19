//! A bad-row output mutant for the prefill ceil comparison.
use kernel_build::CubinSpec;
use kernel_core::{
    contract::DType,
    precision::quantizer::{BlockQuantization, BlockQuantizeQuery, QuantizeOutput},
};
use kernels_sm120::row::block_quantize::{self, Entry};
pub const FAMILY: &str = "ceil_prefill_status_mutant";
pub const CUBIN: &str = "ceil_prefill_status_mutant_sm120";
pub fn entry() -> Entry {
    Entry::new(
        BlockQuantizeQuery {
            width: 1280,
            input: DType::F32,
            format: BlockQuantization::E4M3_CEIL32,
            output: QuantizeOutput::Codes,
        },
        8192,
        "chop_ceil_prefill_ignore_row_flags_v1",
    )
    .unwrap()
    .prefill(8)
    .unwrap()
}
pub fn cubins() -> Vec<CubinSpec> {
    let mut spec = block_quantize::cubin_spec(CUBIN, &[entry()]).unwrap();
    spec.defines
        .push(("CHOP_CEIL_PREFILL_IGNORE_ROW_FLAGS".into(), "1".into()));
    crate::declared(&[CUBIN], vec![spec])
}
