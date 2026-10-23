//! The certified prefill projection (`cuda/gemm/exact_f32_certified.cuh`):
//! the zero-component prefill's single chain over rows a producing norm
//! certified, read as its BF16 output, without the prepare launch. Bitwise
//! the zero-component prefill wherever the F32 activation is that BF16
//! output widened. No catalog or backend selects it; a bundle binds it
//! directly over the norm it runs after.
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

/// Lab: the certified projection with `features`-wide tiles
/// (`gemm/exact_f32_certified_wide.cuh`, which the cubin must include).
pub fn wide_symbol(entry: &Entry, tile_rows: u32, features: u32) -> String {
    format!(
        "{}_certified_m{tile_rows}_f{features}_v1",
        entry.baseline_symbol()
    )
}

/// Lab: the wide tiles' grid, block and shared memory: the served plan's
/// rows and block, `features` weight rows staged per tile, and only the
/// shared memory the certified tiles use (the served plan also reserves
/// the zero path's other two component planes).
pub fn wide_plan(
    entry: &Entry,
    tile_rows: u32,
    features: u32,
    rows: u32,
) -> Result<LaunchPlan, Refusal> {
    if features == 0 || !features.is_multiple_of(32) {
        return Err(Refusal("wide certified tiles take whole 32-feature groups"));
    }
    let mut result = plan(entry, tile_rows, rows)?;
    let tensor = 2 * (features + tile_rows) * 64 * 2;
    let simt = 2 * (tile_rows * 36 + 32 * 33) * 4;
    result.grid[0] = entry.query().n.div_ceil(features);
    result.dynamic_shared_bytes = tensor.max(simt);
    crate::backend::check_plan(&result)?;
    Ok(result)
}

pub fn wide_entry_spec(entry: &Entry, tile_rows: u32, features: u32) -> Result<EntrySpec, Refusal> {
    wide_spec(entry, tile_rows, features, false)
}

/// Lab: the wide tiles whose SIMT path widens the norm's BF16 rows itself
/// and never reads `activation`, which may then be null; the norm need not
/// write its widened rows.
pub fn normed_symbol(entry: &Entry, tile_rows: u32, features: u32) -> String {
    format!(
        "{}_certified_m{tile_rows}_f{features}_normed_v1",
        entry.baseline_symbol()
    )
}

pub fn normed_entry_spec(
    entry: &Entry,
    tile_rows: u32,
    features: u32,
) -> Result<EntrySpec, Refusal> {
    wide_spec(entry, tile_rows, features, true)
}

fn wide_spec(
    entry: &Entry,
    tile_rows: u32,
    features: u32,
    from_normed: bool,
) -> Result<EntrySpec, Refusal> {
    wide_plan(entry, tile_rows, features, 1)?;
    let q = entry.query();
    let mut spec = entry_spec(entry, tile_rows)?;
    spec.symbol = if from_normed {
        normed_symbol(entry, tile_rows, features)
    } else {
        wide_symbol(entry, tile_rows, features)
    };
    spec.body = format!(
        "chop::exact_f32::certified_wide::run<{},{},{},{},{}>(activation,normed,certified,weight,output,live_rows,rows,weight_domain);",
        q.n, q.k, tile_rows, features, from_normed
    );
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery};
    use kernel_core::precision::{ExecutionPhase, OperationUse};

    fn router() -> Entry {
        Entry::new(
            GemmQuery {
                m_capacity: 8192,
                n: 384,
                k: 5120,
                weight: WeightFormat::Bf16,
                activation: ActivationFormat::F32,
                output: DType::F32,
                epilogues: Epilogues::default(),
                grouped: None,
            },
            OperationUse {
                phase: ExecutionPhase::Prefill,
                capacity: 8192,
                graph: false,
            },
            "chop_test_dense_f32_m8192_n384_k5120",
        )
        .unwrap()
    }

    #[test]
    fn wide_tiles_keep_the_served_plan_but_the_feature_grid() {
        let e = router();
        // The served plan reserves the zero path's three component planes;
        // the certified tiles stage one.
        let served = plan(&e, 32, 8192).unwrap();
        let narrow = wide_plan(&e, 32, 32, 8192).unwrap();
        assert_eq!((narrow.grid, narrow.block), (served.grid, served.block));
        assert_eq!(
            (served.dynamic_shared_bytes, narrow.dynamic_shared_bytes),
            (32_768, 17_664)
        );
        for (features, grid, shared) in [(64, 6, 24_576), (128, 3, 40_960)] {
            let p = wide_plan(&e, 32, features, 8192).unwrap();
            assert_eq!(p.grid, [grid, 256, 1]);
            assert_eq!(p.block, [128, 1, 1]);
            assert_eq!(p.dynamic_shared_bytes, shared);
            let spec = wide_entry_spec(&e, 32, features).unwrap();
            assert_eq!(spec.symbol, wide_symbol(&e, 32, features));
            assert_eq!(spec.params, entry_spec(&e, 32).unwrap().params);
            assert!(
                spec.body
                    .contains(&format!("run<384,5120,32,{features},false>"))
            );
            let normed = normed_entry_spec(&e, 32, features).unwrap();
            assert_eq!(
                normed.symbol,
                format!("{}_normed_v1", spec.symbol.trim_end_matches("_v1"))
            );
            assert_eq!(normed.params, spec.params);
            assert!(
                normed
                    .body
                    .contains(&format!("run<384,5120,32,{features},true>"))
            );
        }
        assert!(wide_plan(&e, 32, 48, 8192).is_err());
    }
}
