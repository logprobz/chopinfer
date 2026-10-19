//! Plain input norm and sticky-status quantizer after a lagged collapse.
use kernel_build::CubinSpec;
use kernel_core::contract::{DType, RmsNormKind};
use kernel_core::precision::quantizer::{BlockQuantization, BlockQuantizeQuery, QuantizeOutput};
use kernel_core::rmsnorm::RmsNormRequirement;
use kernels_sm120::row::{
    block_quantize,
    rmsnorm::{RmsNorm, RmsNormTarget},
};
use kernels_sm120::sinkhorn_hc::lagged::LAB;

pub const FAMILY: &str = "sinkhorn_hc_lagged_boundary";
pub const CUBIN: &str = "sinkhorn_hc_lagged_boundary_sm120";
pub const NORM: &str = "chop_hc_lagged_boundary_norm_v1";

pub fn norm() -> RmsNorm {
    RmsNorm::new(
        RmsNormRequirement::bf16_backend_chosen(
            LAB.mix.width,
            f32::from_bits(LAB.mix.norm_eps_bits),
            RmsNormKind::Plain,
            false,
        ),
        LAB.capacity,
        RmsNormTarget::SM120F,
    )
    .unwrap()
}

pub fn quantize() -> block_quantize::Entry {
    block_quantize::Entry::new(
        BlockQuantizeQuery {
            width: LAB.mix.width,
            input: DType::Bf16,
            output: QuantizeOutput::Codes,
            format: BlockQuantization::E4M3Ceil32 {
                amax_floor_bits: 1e-4_f32.to_bits(),
            },
        },
        LAB.capacity,
        "chop_hc_lagged_boundary_quantize_v1",
    )
    .unwrap()
}

pub fn cubins() -> Vec<CubinSpec> {
    let mut spec = norm().cubin_spec(CUBIN, NORM);
    spec.headers
        .push(format!("#include \"{}\"", block_quantize::HEADER));
    spec.entries.push(quantize().spec());
    crate::declared(&[CUBIN], vec![spec])
}
