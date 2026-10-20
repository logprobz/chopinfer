//! Unselected certified prefill and decode projections with device controls.
use kernel_build::CubinSpec;
use kernels_sm120::{
    f32_projection::{Entry, zero_decode, zero_prefill},
    row::rmsnorm::{RmsNorm, RmsNormTarget},
};
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "projection_zero";
pub const CUBIN: &str = "projection_zero_sm120";
pub fn entries() -> Vec<Entry> {
    bundle_deepseek41::sm120::f32_prefill::baseline_entries(&FLASH)
        .unwrap()
        .into_iter()
        .filter(|e| [32, 384, 512].contains(&e.query().n) && e.query().k == 5120)
        .collect()
}
pub fn entry() -> Entry {
    let n: u32 = std::env::var("CHOPINFER_PROJECTION_ZERO_N")
        .unwrap_or_else(|_| "512".into())
        .parse()
        .unwrap();
    entries()
        .into_iter()
        .find(|e| e.query().n == n)
        .expect("declared shape")
}
pub fn control(e: &Entry, name: &str) -> String {
    format!("chop_projection_zero_n{}_{name}_v1", e.query().n)
}
pub fn norm() -> RmsNorm {
    RmsNorm::new(
        model_deepseek41_contract::rows::norm_quantizers(&FLASH)[0].norm,
        8192,
        RmsNormTarget::SM120F,
    )
    .unwrap()
}
pub fn cubins() -> Vec<CubinSpec> {
    let entries = entries();
    let mut spec = zero_prefill::cubin_spec(CUBIN, &entries).unwrap();
    for e in &entries {
        let mut bad = zero_prefill::entry_spec(e, 32).unwrap();
        bad.symbol = control(e, "omit_chain");
        bad.body = bad.body.replace(",32>(", ",32,true>(");
        spec.entries.push(bad);
        let mut prep = spec
            .entries
            .iter()
            .find(|s| s.symbol == zero_prefill::prepare_symbol(e))
            .unwrap()
            .clone();
        prep.symbol = control(e, "force_certificate");
        prep.body = prep.body.replace("prepare<5120>", "prepare<5120,true>");
        spec.entries.push(prep);
        let decode = zero_decode::entry_spec(e).unwrap();
        let mut bad = decode.clone();
        bad.symbol = control(e, "force_decode");
        bad.body = bad.body.replace("false,true>", "false,true,true>");
        spec.entries.extend([decode, bad]);
    }
    let norm = norm();
    spec.headers.extend(norm.headers());
    spec.entries
        .push(norm.entry("chop_projection_zero_replay_norm_v1"));
    vec![spec]
}
