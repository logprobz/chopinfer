//! The SM120 facts the fake-host tests check launches, journals and loaded
//! modules against: the HC precompute module's and mixes' names, the
//! parallel route's ticket bytes, the encoder catalog's entries and cubins,
//! and the kernels' invariance declarations.

pub(super) use kernels_sm120::gemm::fused::{
    HC_PADDED_MIX, HC_PRECOMPUTE_CUBIN, HC_PRECOMPUTED_MIX,
};
pub(super) use kernels_sm120::invariance::{Declared, Launched, OpKind, Program};
pub(super) use kernels_sm120::moe::parallel::PARALLEL_ROUTE_TICKET_BYTES;

/// The full symbol of the encoder entry whose symbol reads `stem`.
pub(super) fn encoder_symbol(stem: &str) -> &'static str {
    kernels_sm120::encoder::by_stem(crate::encoder::kernels().unwrap(), stem)
        .unwrap()
        .symbol
        .as_str()
}

/// The encoder catalog's cubins.
pub(super) fn encoder_cubins() -> [kernel_build::CubinSpec; 2] {
    kernels_sm120::encoder::cubin_specs(crate::encoder::kernels().unwrap()).unwrap()
}
