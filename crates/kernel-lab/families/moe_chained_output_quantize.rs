//! Shared ceil quantizers consumed by routed gate/up conformance.
use kernel_build::CubinSpec;
use kernel_core::contract::DType;
use kernel_core::precision::quantizer::{BlockQuantization, BlockQuantizeQuery, QuantizeOutput};
use kernels_sm120::row::block_quantize::{self, Entry};
pub const FAMILY: &str = "moe_chained_output_quantize";
pub const CUBIN: &str = "chop_moe_chain_output_quantize_lab";
pub fn entries() -> Vec<Entry> {
    [3, 6]
        .into_iter()
        .map(|top| {
            Entry::new(
                BlockQuantizeQuery {
                    width: top * 576,
                    input: DType::Bf16,
                    format: BlockQuantization::E4M3_CEIL32,
                    output: QuantizeOutput::Codes,
                },
                32,
                format!("chop_moe_chain_output_ceil_top{top}_v1"),
            )
            .unwrap()
        })
        .collect()
}
pub fn cubins() -> Vec<CubinSpec> {
    vec![block_quantize::cubin_spec(CUBIN, &entries()).unwrap()]
}
