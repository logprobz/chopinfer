//! The shared expert's activation the bundle registers on SM120 and its
//! build compiles: the clamped SwiGLU with one BF16 rounding and the ceil
//! E4M3/32 quantization (`kernels_sm120::row::silu_mul_quantize`) at each
//! supported rank count's slice of the intermediate. Its gate, up and down
//! projections are dense fold GEMMs.

use kernel_build::CubinSpec;
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::silu_mul_quantize::{self, Entry};
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::moe;
use model_deepseek41_contract::serving::PREFILL_TOKENS;

pub const FAMILY: &str = "shared_expert";

/// The name of `g`'s activation cubin.
pub fn cubin_name(g: &Geometry) -> String {
    format!("shared_{}_sm120", g.kernel_tag)
}

/// The activation of `g` at `ranks` ranks.
pub fn activation(g: &Geometry, ranks: u32) -> Result<Entry, String> {
    check_ranks(g, ranks)?;
    let query = moe::shared_activation(g, ranks);
    Entry::new(
        query,
        PREFILL_TOKENS,
        format!("chop_{}_shared_swiglu_w{}_v1", g.kernel_tag, query.width),
    )
    .map_err(|e| e.to_string())
}

/// Each geometry's activation cubin, with every supported rank count's
/// width.
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = Vec::new();
    for g in geometries {
        let mut entries: Vec<Entry> = Vec::new();
        for &ranks in SUPPORTED_RANKS {
            if check_ranks(g, ranks).is_err() {
                continue;
            }
            let entry = activation(g, ranks)?;
            if !entries.iter().any(|e| e.symbol == entry.symbol) {
                entries.push(entry);
            }
        }
        specs.push((
            FAMILY,
            silu_mul_quantize::cubin_spec(&cubin_name(g), &entries).map_err(|e| e.to_string())?,
        ));
    }
    Ok(specs)
}

pub fn register(backend: Sm120, g: &Geometry, ranks: u32) -> Result<Sm120, String> {
    Ok(backend.with_silu_mul_quantize(activation(g, ranks)?))
}
