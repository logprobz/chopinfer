//! Native latent glue and its shared output quantizer, independent of models.
use kernel_build::CubinSpec;
use kernel_core::contract::DType;
use kernel_core::precision::quantizer::{BlockQuantization, BlockQuantizeQuery, QuantizeOutput};
use kernels_sm120::latent::glue::{self, Entry, Kernel};
use kernels_sm120::row::block_quantize;

pub const FAMILY: &str = "latent_glue";
pub const CAPACITY: u32 = 320;
pub fn entries() -> Vec<Entry> {
    [
        (Kernel::QueryRope, "query_rope"),
        (Kernel::WindowRotate, "window_rope"),
        (Kernel::WindowStore, "window_store"),
        (Kernel::WindowKeys, "window_keys"),
    ]
    .into_iter()
    .map(|(kernel, tag)| Entry {
        kernel,
        capacity: CAPACITY,
        symbol: format!("chop_latent_glue_{tag}_v1"),
    })
    .collect()
}
pub fn quantizers() -> Vec<block_quantize::Entry> {
    vec![
        block_quantize::Entry::new(
            BlockQuantizeQuery {
                width: 512,
                input: DType::F32,
                format: BlockQuantization::E4M3_CEIL32,
                output: QuantizeOutput::Codes,
            },
            CAPACITY,
            "chop_latent_glue_ceil32_v1",
        )
        .unwrap(),
    ]
}
pub fn cubins() -> Vec<CubinSpec> {
    vec![
        glue::cubin_spec("chop_latent_glue_lab", &entries()),
        block_quantize::cubin_spec("chop_latent_glue_quantize_lab", &quantizers()).unwrap(),
    ]
}
