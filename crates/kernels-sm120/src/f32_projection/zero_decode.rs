//! Opt-in decode compute with a warp-local exact-BF16 certificate.
//! The cached row admission, partial layout and ordered reducer are unchanged.
use super::Entry;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

pub fn symbol(entry: &Entry) -> String {
    format!("{}_zero_decode_v1", entry.baseline_symbol())
}

fn cached(entry: &Entry) -> Result<Entry, Refusal> {
    if entry.query().weight != WeightFormat::Bf16 {
        return Err(Refusal("zero-component decode requires BF16 weights"));
    }
    Ok(Entry::new(entry.query(), entry.usage(), entry.baseline_symbol())?.with_decode())
}

pub fn plan(entry: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    cached(entry)?.decode_plan(rows)
}

pub fn entry_spec(entry: &Entry) -> Result<EntrySpec, Refusal> {
    let cached = cached(entry)?;
    let mut spec = cached
        .entry_specs()
        .into_iter()
        .find(|s| s.symbol == cached.stage_symbol("_decode"))
        .expect("cached decode compute");
    spec.symbol = symbol(entry);
    spec.body = format!(
        "chop::exact_f32::decode<{},{},false,true>(activation,weight,workspace+32,live_rows,rows,weight_domain,reinterpret_cast<const uint32_t*>(workspace));",
        entry.query().n,
        entry.query().k
    );
    Ok(spec)
}

pub fn cubin_spec(name: impl Into<String>, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut spec = super::cubin_spec(name, entries)?;
    spec.entries = entries.iter().map(entry_spec).collect::<Result<_, _>>()?;
    spec.headers
        .push("#include \"gemm/exact_f32_decode.cuh\"".into());
    Ok(spec)
}
