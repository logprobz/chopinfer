//! Prefill candidates with shared panels and independent output registers.
//! The caller qualifies exact row counts before adding serving dispatch.
use super::Entry;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

pub fn symbol(entry: &Entry, tile_rows: u32) -> String {
    format!("{}_blocked_m{tile_rows}_v1", entry.baseline_symbol())
}

pub fn plan(entry: &Entry, tile_rows: u32, rows: u32) -> Result<LaunchPlan, Refusal> {
    let q = entry.query();
    if !matches!(tile_rows, 32 | 64) || q.k % 128 != 0 || rows == 0 || rows > q.m_capacity {
        return Err(Refusal(
            "blocked F32 requires 32/64-row tiles and complete 128-K chains",
        ));
    }
    let tensor = 2 * (32 + 3 * tile_rows) * 64 * 2;
    let simt = 2 * (tile_rows * 36 + 32 * 33) * 4;
    let result = LaunchPlan {
        grid: [q.n.div_ceil(32), rows.div_ceil(tile_rows), 1],
        block: [128, 1, 1],
        dynamic_shared_bytes: tensor.max(simt),
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    crate::backend::check_plan(&result)?;
    Ok(result)
}

pub fn entry_spec(entry: &Entry, tile_rows: u32) -> Result<EntrySpec, Refusal> {
    plan(entry, tile_rows, 1)?;
    let q = entry.query();
    let mut spec = entry.clone().with_prefill_tiles(32)?.entry_spec();
    spec.symbol = symbol(entry, tile_rows);
    spec.body = format!(
        "chop::exact_f32::blocked::run<{},{},{},{}>(activation,weight,workspace,output,live_rows,rows,weight_domain);",
        q.n,
        q.k,
        tile_rows,
        q.weight == WeightFormat::F32
    );
    spec.max_threads = 128;
    Ok(spec)
}

pub fn cubin_spec(name: impl Into<String>, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let staged = entries
        .iter()
        .cloned()
        .map(|e| e.with_prefill_tiles(32))
        .collect::<Result<Vec<_>, _>>()?;
    let mut spec = super::cubin_spec(name, &staged)?;
    spec.headers
        .push("#include \"gemm/exact_f32_blocked.cuh\"".into());
    spec.entries
        .retain(|e| !staged.iter().any(|s| s.symbol() == e.symbol));
    for entry in entries {
        for rows in [32, 64] {
            spec.entries.push(entry_spec(entry, rows)?);
        }
    }
    Ok(spec)
}
