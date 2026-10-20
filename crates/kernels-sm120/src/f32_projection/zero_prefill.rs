//! Prefill specialization for rows certified to be exact widened BF16.
//! General F32 and out-of-domain rows retain their original arithmetic.
use super::{Entry, blocked};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

pub fn workspace_bytes(entry: &Entry, rows: u32) -> u64 {
    entry.prefill_workspace_bytes(rows) + u64::from(rows) * 8
}
pub fn symbol(entry: &Entry, tile_rows: u32) -> String {
    format!("{}_zero_m{tile_rows}_v1", entry.baseline_symbol())
}
pub fn prepare_symbol(entry: &Entry) -> String {
    format!("{}_zero_prepare_v1", entry.baseline_symbol())
}
pub fn plan(entry: &Entry, tile_rows: u32, rows: u32) -> Result<LaunchPlan, Refusal> {
    if entry.query().weight != WeightFormat::Bf16 {
        return Err(Refusal("zero-component prefill requires BF16 weights"));
    }
    blocked::plan(entry, tile_rows, rows)
}
pub fn entry_spec(entry: &Entry, tile_rows: u32) -> Result<EntrySpec, Refusal> {
    plan(entry, tile_rows, 1)?;
    let q = entry.query();
    let mut spec = blocked::entry_spec(entry, tile_rows)?;
    spec.symbol = symbol(entry, tile_rows);
    spec.body = format!(
        "chop::exact_f32::zero_prefill::run<{},{},{}>(activation,weight,workspace,output,live_rows,rows,weight_domain);",
        q.n, q.k, tile_rows
    );
    Ok(spec)
}
pub fn cubin_spec(name: impl Into<String>, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut spec = blocked::cubin_spec(name, entries)?;
    spec.headers
        .push("#include \"gemm/exact_f32_zero.cuh\"".into());
    spec.entries.retain(|s| {
        entries.iter().any(|e| {
            s.symbol
                == e.clone()
                    .with_prefill_tiles(32)
                    .unwrap()
                    .stage_symbol("_weights")
        })
    });
    for e in entries {
        let prepared = e.clone().with_prefill_tiles(32)?;
        let mut prep = prepared
            .prefill_specs()
            .into_iter()
            .find(|s| s.symbol == prepared.stage_symbol("_prepare"))
            .unwrap();
        prep.symbol = prepare_symbol(e);
        prep.body = format!(
            "chop::exact_f32::zero_prefill::prepare<{}>(activation,live_rows,rows,workspace);",
            e.query().k
        );
        spec.entries.push(prep);
        for m in [32, 64] {
            spec.entries.push(entry_spec(e, m)?);
        }
    }
    Ok(spec)
}
