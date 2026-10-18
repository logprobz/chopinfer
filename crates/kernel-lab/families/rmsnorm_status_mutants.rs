//! Dropped-status controls for F32 RMSNorm.
use kernel_build::CubinSpec;
use kernel_core::rmsnorm::RmsNormRequirement;
use kernels_sm120::row::rmsnorm::{RmsNorm, RmsNormTarget};
pub const FAMILY: &str = "rmsnorm_status_mutants";
pub fn entries() -> Vec<(RmsNormRequirement,String)> {
    crate::rmsnorm_f32::WIDTHS.into_iter().map(|w| (crate::rmsnorm_f32::requirement(w),format!("chop_rmsnorm_drop_flag_w{w}_v1"))).collect()
}
pub fn cubins() -> Vec<CubinSpec> {
    let entries=entries();
    let norm=|q| RmsNorm::new(q,32,RmsNormTarget::SM120F).unwrap();
    let mut spec=norm(entries[0].0).cubin_spec("rmsnorm_status_mutants_sm120",&entries[0].1);
    spec.headers.push("#include \"latent/glue_mutants.cuh\"".into());
    spec.entries=entries.into_iter().map(|(q,symbol)| {
        let mut e=norm(q).entry(symbol);
        e.body=e.body.replace(&format!("rmsnorm_f32<{}>",q.width),&format!("rmsnorm_f32<{},chop::latent::glue::DropFlags>",q.width));e
    }).collect();vec![spec]
}
