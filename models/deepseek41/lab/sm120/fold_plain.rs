//! The served fold cubins whose shared rows the bundle swizzles, built with
//! `CHOP_FOLD_PLAIN_ROWS`: the unswizzled placement, as the lab reference for
//! the served entries. Each cubin is the bundle's spec with a `plain_` name
//! prefix and the define; its symbols are the served symbols.
use bundle_deepseek41::sm120::{fold_local_rows, mxfp8_fold, projection_reuse};
use kernel_build::CubinSpec;
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "fold_plain";
pub const PREFIX: &str = "plain_";
pub const DEFINE: &str = "CHOP_FOLD_PLAIN_ROWS";
/// The bundle's lock family and spec of every served cubin this family
/// rebuilds (of the row-local cubin, the pair-loop entries only).
pub fn served() -> Vec<(&'static str, CubinSpec)> {
    // Pin the historical row-major recipes even after the bundle constructors
    // replace their readers with packed twins. Existing lab locks stay put.
    let mut specs = vec![
        (
            mxfp8_fold::TICKET_FAMILY,
            kernels_sm120::gemm::mxfp8_fold_ticket::cubin_spec(
                &mxfp8_fold::ticket_cubin_name(&FLASH),
                &mxfp8_fold::ticket_catalog(&FLASH).unwrap(),
            )
            .unwrap(),
        ),
        (
            mxfp8_fold::PREFETCH_FAMILY,
            kernels_sm120::gemm::mxfp8_fold_ticket::cubin_spec(
                &mxfp8_fold::prefetch_cubin_name(&FLASH),
                &mxfp8_fold::prefetch_catalog(&FLASH).unwrap(),
            )
            .unwrap(),
        ),
        (
            mxfp8_fold::DECODE_FAMILY,
            kernels_sm120::gemm::mxfp8_fold_decode::cubin_spec(
                &mxfp8_fold::decode_cubin_name(&FLASH),
                &mxfp8_fold::decode_catalog(&FLASH).unwrap(),
            )
            .unwrap(),
        ),
        (
            projection_reuse::FOLD_FAMILY,
            kernels_sm120::gemm::mxfp8_fold_reuse::cubin_spec(
                projection_reuse::FOLD_CUBIN,
                &projection_reuse::entries().unwrap(),
            )
            .unwrap(),
        ),
    ];
    // The deep rings place rows by their own swizzle flag, so the family
    // rebuilds only the pair loops.
    let pair_loops: Vec<_> = fold_local_rows::entries()
        .into_iter()
        .filter(|e| e.stages == 0)
        .collect();
    specs.push((
        fold_local_rows::FAMILY,
        kernels_sm120::gemm::mxfp8_fold_local_rows::cubin_spec(fold_local_rows::CUBIN, &pair_loops)
            .unwrap(),
    ));
    specs
}
pub fn cubins() -> Vec<CubinSpec> {
    served()
        .into_iter()
        .map(|(_, mut spec)| {
            spec.name = format!("{PREFIX}{}", spec.name);
            spec.defines.push((DEFINE.into(), "1".into()));
            spec
        })
        .collect()
}
