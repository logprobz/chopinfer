//! Wider feature staging with three exact BF16 component chains.
use super::Entry;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

pub fn symbol(entry: &Entry, features: u32) -> String {
    format!("{}_wide_m32_n{features}_v1", entry.baseline_symbol())
}
pub fn plan(entry: &Entry, features: u32, rows: u32) -> Result<LaunchPlan, Refusal> {
    let q = entry.query();
    if !matches!(features, 64 | 128) || q.k % 128 != 0 || rows == 0 || rows > q.m_capacity {
        return Err(Refusal(
            "wide F32 requires N64/N128 tiles and complete 128-K chains",
        ));
    }
    let tensor = 2 * (features + 3 * 32) * 64 * 2;
    let simt = 2 * (32 * 36 + 32 * (features + 1)) * 4;
    let result = LaunchPlan {
        grid: [q.n.div_ceil(features), rows.div_ceil(32), 1],
        block: [features * 4, 1, 1],
        dynamic_shared_bytes: tensor.max(simt),
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    crate::backend::check_plan(&result)?;
    Ok(result)
}
pub fn entry_spec(entry: &Entry, features: u32) -> Result<EntrySpec, Refusal> {
    plan(entry, features, 1)?;
    let q = entry.query();
    let mut spec = entry.clone().with_prefill_tiles(32)?.entry_spec();
    spec.symbol = symbol(entry, features);
    spec.body = format!(
        "chop::exact_f32::wide::run<{},{},32,{},{}>(activation,weight,workspace,output,live_rows,rows,weight_domain);",
        q.n,
        q.k,
        features,
        q.weight == WeightFormat::F32
    );
    spec.max_threads = features * 4;
    Ok(spec)
}
pub fn cubin_spec(name: impl Into<String>, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut spec = super::cubin_spec(name, entries)?;
    spec.headers = vec!["#include \"gemm/exact_f32_wide.cuh\"".into()];
    spec.entries.clear();
    for entry in entries {
        for features in [64, 128] {
            spec.entries.push(entry_spec(entry, features)?);
        }
    }
    Ok(spec)
}
