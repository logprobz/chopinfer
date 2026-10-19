//! Frozen vector-store entries shared by serving and the qualification lab.
pub const FAMILY: &str = "exact_moe_prefill_vector";
pub const CUBIN: &str = "exact_moe_prefill_vector_sm120";
pub const VARIANTS: [&str; 3] = ["pipe64", "wide128", "decoded128"];
pub fn symbol(name: &str) -> String {
    format!("{}_vector_v1", name.strip_suffix("_v1").unwrap())
}
pub fn cubins(mut scalar: Vec<kernel_build::CubinSpec>) -> Vec<kernel_build::CubinSpec> {
    let mut spec = scalar.remove(0);
    spec.name = CUBIN.into();
    spec.headers = vec!["#include \"moe/prefill_pipeline_vector.cuh\"".into()];
    spec.entries.retain(|e| {
        VARIANTS
            .iter()
            .any(|v| e.symbol.contains(&format!("_{v}_")))
    });
    for entry in &mut spec.entries {
        entry.symbol = symbol(&entry.symbol);
        entry.body = entry
            .body
            .replace("prefill_pipeline::", "prefill_pipeline_vector::");
    }
    vec![spec]
}
