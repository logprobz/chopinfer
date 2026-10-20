//! Reuse exact-F32 weight tiles across four ordered eight-row groups.
use super::Entry;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

pub fn symbol(entry: &Entry) -> String {
    format!("{}_reuse32_v1", entry.stage_symbol("_decode"))
}

pub fn plan(entry: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    if !entry.uses_cached_weights()
        || entry.query().weight != WeightFormat::Bf16
        || !crate::gemm::mxfp8_fold_reuse::supported_rows(rows)
    {
        return Err(Refusal(
            "exact weight reuse requires cached BF16 weights and a supported row bucket",
        ));
    }
    let mut result = entry.decode_plan(rows)?;
    result.grid[1] = rows.div_ceil(32);
    Ok(result)
}

pub fn spec(entry: &Entry) -> Result<EntrySpec, Refusal> {
    plan(entry, 8)?;
    let mut result = entry
        .entry_specs()
        .into_iter()
        .find(|s| s.symbol == entry.stage_symbol("_decode"))
        .ok_or(Refusal(
            "exact weight reuse requires the cached decode entry",
        ))?;
    let q = entry.query();
    result.symbol = symbol(entry);
    result.body = format!(
        "__shared__ chop::exact_f32_reuse::Arena arena;const chop::exact_f32_reuse::Task task{{blockIdx.x,blockIdx.y,blockIdx.z}};chop::ptx::griddep_launch_dependents();chop::ptx::griddep_wait();chop::exact_f32_reuse::task<{},{},false>(task,arena,activation,weight,workspace+32,live_rows,rows,weight_domain,reinterpret_cast<const uint32_t*>(workspace));",
        q.n, q.k
    );
    Ok(result)
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("exact weight reuse requires entries"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/exact_f32_reuse.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(spec).collect::<Result<_, _>>()?,
    })
}

impl Entry {
    /// Opt into the compiled reuse entry only at explicit measured row counts.
    pub fn with_decode_reuse(mut self, mut rows: Vec<u32>) -> Result<Self, Refusal> {
        rows.sort_unstable();
        if rows.is_empty() || rows.windows(2).any(|w| w[0] == w[1]) {
            return Err(Refusal("decode reuse needs distinct measured row buckets"));
        }
        for &row in &rows {
            plan(&self.entry_for_rows(row), row)?;
        }
        self.reuse_rows = rows;
        Ok(self)
    }

    pub fn uses_decode_reuse(&self, rows: u32) -> bool {
        self.reuse_rows.binary_search(&rows).is_ok()
    }
}
