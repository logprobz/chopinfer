//! The certified prefill projection (`cuda/gemm/exact_f32_certified.cuh`):
//! the zero-component prefill's single chain over rows a producing norm
//! certified, read as its BF16 output, without the prepare launch. Bitwise
//! the zero-component prefill wherever the F32 activation is that BF16
//! output widened. No catalog or backend selects it yet.
use super::{Entry, blocked, pointer};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

pub fn symbol(entry: &Entry, tile_rows: u32) -> String {
    format!("{}_certified_m{tile_rows}_v1", entry.baseline_symbol())
}

/// The zero-component prefill's grid, block and shared memory.
pub fn plan(entry: &Entry, tile_rows: u32, rows: u32) -> Result<LaunchPlan, Refusal> {
    if entry.query().weight != WeightFormat::Bf16 {
        return Err(Refusal("certified prefill requires BF16 weights"));
    }
    blocked::plan(entry, tile_rows, rows)
}

pub fn entry_spec(entry: &Entry, tile_rows: u32) -> Result<EntrySpec, Refusal> {
    plan(entry, tile_rows, 1)?;
    let q = entry.query();
    Ok(EntrySpec::new(
        symbol(entry, tile_rows),
        vec![
            pointer("activation", "const float"),
            pointer("normed", "const uint16_t"),
            pointer("certified", "const uint32_t"),
            pointer("weight", "const uint16_t"),
            pointer("output", "float"),
            pointer("live_rows", "const uint32_t"),
            EntryParam::U32 {
                name: "rows".into(),
            },
            pointer("weight_domain", "const uint32_t"),
        ],
        format!(
            "chop::exact_f32::certified::run<{},{},{}>(activation,normed,certified,weight,output,live_rows,rows,weight_domain);",
            q.n, q.k, tile_rows
        ),
        128,
        1,
    ))
}

/// The certified entries of `entries` at `tile_rows`, one cubin.
pub fn cubin_spec(
    name: impl Into<String>,
    entries: &[Entry],
    tile_rows: &[u32],
) -> Result<CubinSpec, Refusal> {
    let mut specs = Vec::new();
    for e in entries {
        for &m in tile_rows {
            specs.push(entry_spec(e, m)?);
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: Vec::new(),
        headers: vec!["#include \"gemm/exact_f32_certified.cuh\"".into()],
        defines: Vec::new(),
        entries: specs,
    })
}
