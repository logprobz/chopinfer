//! The shared row entries the bundle registers on SM120 and its build
//! compiles: the RMSNorms of every width the program normalizes and the
//! vocabulary-sharded embedding gather of every rank count, one cubin per
//! geometry, and the shared row copies' cubin (the stream broadcast).
//! The contracts are the model's (`model_deepseek41_contract::rows`).

use kernel_build::{CubinSpec, EntrySpec};
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::RowInstantiation;
use kernels_sm120::row::rmsnorm::{RmsNorm, RmsNormTarget};
use kernels_sm120::{Profile, ProfileNames};
use model_deepseek41_contract::admission::SUPPORTED_RANKS;
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::rows;
use model_deepseek41_contract::serving::PREFILL_TOKENS;

/// The lock family of the row cubins, one per geometry.
pub const FAMILY: &str = "rows";
/// The lock family of the shared row copies' cubin.
pub const COPY_FAMILY: &str = "row_copy";
pub const COPY_CUBIN: &str = "row_copy_sm120";

/// The kernel profile of `g`: every entry symbol carries its kernel tag.
pub const fn profile(g: &Geometry) -> Profile {
    Profile::synthetic(ProfileNames {
        name: "deepseek41",
        tag: g.kernel_tag,
    })
}

pub fn norm_symbol(g: &Geometry, width: u32) -> String {
    format!("chop_row_rmsnorm_{}_w{width}_v1", g.kernel_tag)
}

/// The embedding gather of one rank's vocabulary shard at `ranks` ranks:
/// rows of other shards are zero, which the all-reduce fills.
pub fn embedding(g: &Geometry, ranks: u32) -> RowInstantiation {
    let mut entry = kernels_sm120::row::embedding(
        profile(g),
        &format!("embedding_{}_tp{ranks}", g.kernel_tag),
        g.shape.vocab / ranks,
        g.shape.hidden,
        true,
    );
    entry.invariance = kernels_sm120::row::row_local();
    entry
}

/// The rank counts `g` serves.
fn ranks(g: &Geometry) -> impl Iterator<Item = u32> + '_ {
    SUPPORTED_RANKS
        .iter()
        .copied()
        .filter(|&ranks| model_deepseek41_contract::admission::check_ranks(g, ranks).is_ok())
}

/// `g`'s norm widths (`rows::norm_widths`), each once, in their first
/// order: two of them may be equal at any geometry, not only adjacent ones.
fn norm_widths(g: &Geometry) -> Vec<u32> {
    unique(&rows::norm_widths(g))
}

fn unique(widths: &[u32]) -> Vec<u32> {
    let mut unique = Vec::with_capacity(widths.len());
    for &width in widths {
        if !unique.contains(&width) {
            unique.push(width);
        }
    }
    unique
}

/// `g`'s row cubin: its norms, planned at a prefill's rows, and the
/// embedding gather of each rank count.
pub fn cubin(g: &Geometry) -> Result<CubinSpec, String> {
    let mut entries = Vec::new();
    let mut headers = Vec::new();
    for width in norm_widths(g) {
        let norm = RmsNorm::new(rows::norm(g, width), PREFILL_TOKENS, RmsNormTarget::SM120F)
            .map_err(|refusal| format!("RMSNorm of {width}: {refusal}"))?;
        for header in norm.headers() {
            if !headers.contains(&header) {
                headers.push(header);
            }
        }
        entries.push(norm.entry(norm_symbol(g, width)));
    }
    for ranks in ranks(g) {
        let embedding = embedding(g, ranks);
        entries.push(EntrySpec::new(
            embedding.symbol,
            embedding.params,
            embedding.body,
            embedding.threads,
            1,
        ));
    }
    Ok(CubinSpec {
        name: format!("row_{}_sm120", g.kernel_tag),
        sources_root: kernels_sm120::cuda_source_root(),
        include_dirs: vec![],
        headers,
        defines: vec![],
        entries,
    })
}

/// Every row cubin of `geometries`, with its lock family, then the shared
/// row copies'.
pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    let mut specs = geometries
        .iter()
        .map(|g| Ok((FAMILY, cubin(g)?)))
        .collect::<Result<Vec<_>, String>>()?;
    specs.push((
        COPY_FAMILY,
        kernels_sm120::row::copy::cubin_spec(COPY_CUBIN, &kernels_sm120::row::copy::entries()),
    ));
    Ok(specs)
}

/// `sm120` with `g`'s row entries at `ranks` ranks registered.
pub fn register(mut sm120: Sm120, g: &Geometry, ranks: u32) -> Sm120 {
    for width in norm_widths(g) {
        sm120 = sm120.with_rmsnorm(rows::norm(g, width), norm_symbol(g, width));
    }
    sm120 = sm120.with_sharded_embedding(embedding(g, ranks));
    kernels_sm120::row::copy::entries()
        .into_iter()
        .fold(sm120, |sm120, entry| sm120.with_row_copy(entry))
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES};

    /// Equal widths anywhere in the list register one norm, in first order;
    /// Flash keeps its order, so its lock bytes stay.
    #[test]
    fn each_norm_width_registers_once_in_first_order() {
        assert_eq!(unique(&[512, 256, 512]), [512, 256]);
        assert_eq!(unique(&[512, 512, 256]), [512, 256]);
        assert_eq!(norm_widths(&FLASH), [5120, 1280, 512]);
        for g in GEOMETRIES {
            let cubin = cubin(g).unwrap();
            let mut symbols: Vec<&str> = cubin.entries.iter().map(|e| e.symbol.as_str()).collect();
            let count = symbols.len();
            symbols.sort_unstable();
            symbols.dedup();
            assert_eq!(symbols.len(), count, "{}", g.id);
        }
    }
}
