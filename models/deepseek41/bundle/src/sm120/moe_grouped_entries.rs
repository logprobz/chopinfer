//! Qualified grouped down entry. Arithmetic and symbols match the paired lab entry.
use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::moe::{
    chained::quantized::Projection,
    grouped_bf16::{Entry, Tile},
};
use model_deepseek41_contract::{geometry::FLASH, moe};
pub const FAMILY: &str = "grouped_bf16_down_prefill";
pub const CUBIN: &str = "grouped_bf16_down_prefill_deepseek41_sm120";
pub fn entry() -> Entry {
    Entry::new(
        Projection::new(
            moe::projection(
                &FLASH,
                4,
                moe::Experts {
                    count: 384,
                    top_k: 6,
                },
                GatedMlpProjection::Down,
                8192,
            )
            .unwrap(),
            64,
            8192,
        )
        .unwrap(),
        Tile {
            rows: 128,
            columns: 128,
            stages: 2,
            whole_chain: true,
        },
    )
    .unwrap()
}
pub fn cubin() -> kernel_build::CubinSpec {
    kernels_sm120::moe::grouped_bf16::cubin(CUBIN, &[entry()], false)
}
