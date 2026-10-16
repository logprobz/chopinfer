//! Lab-only cubins over MiMo V2.6 Flash's entries, which kernel-lab
//! builds beside the served ones: 64-row scale probes.

use kernel_build::CubinSpec;
use kernel_core::contract::{StaticQuery, WeightFormat};
use kernels_sm120::gemm::{self, GemmVariant};

/// Lab-only cubin: TP4's QKV shapes (decode buckets and prefill) with
/// 64-row FP8 scales, symbols `..._rows64_v1`. With each 128-row scale
/// repeated for its two 64-row groups, every entry must reproduce its TP4
/// block-scaled entry bit for bit.
pub fn rows64_probe_catalog() -> Vec<GemmVariant> {
    super::geometry::require_base_geometries("kernel-lab")
        .unwrap_or_else(|error| panic!("{error}"));
    super::gemm::catalog()
        .into_iter()
        .filter(|item| {
            item.profile == crate::sm120::config::PROFILE
                && item.symbol.starts_with("chop_gemm_mimo_qkv_")
        })
        .map(|mut item| {
            if let StaticQuery::Gemm(mut query) = item.instantiation.key.query {
                query.weight =
                    WeightFormat::Fp8Rows64x128(kernel_core::contract::ScaleMode::ExactF32);
                item.instantiation.key.query = query.into();
            }
            item.instantiation.symbol = item.instantiation.symbol.replace("_v1", "_rows64_v1");
            item
        })
        .collect()
}

pub fn rows64_probe_cubin_spec() -> CubinSpec {
    let entries = rows64_probe_catalog()
        .into_iter()
        .map(|item| item.instantiation)
        .collect::<Vec<_>>();
    let mut spec = gemm::cubin_spec(&entries).expect("64-row scale probe spec");
    spec.name = "gemm_rows64_probe_sm120".into();
    spec
}
