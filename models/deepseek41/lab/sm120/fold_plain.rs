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
/// rebuilds.
pub fn served() -> Vec<(&'static str, CubinSpec)> {
    let g = [FLASH];
    let mut specs = Vec::new();
    specs.extend(mxfp8_fold::ticket_cubins(&g).unwrap());
    specs.extend(mxfp8_fold::prefetch_cubins(&g).unwrap());
    specs.extend(mxfp8_fold::decode_cubins(&g).unwrap());
    specs.extend(
        projection_reuse::cubins(&g)
            .unwrap()
            .into_iter()
            .filter(|(family, _)| *family == projection_reuse::FOLD_FAMILY),
    );
    specs.extend(fold_local_rows::cubins(&g).unwrap());
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
