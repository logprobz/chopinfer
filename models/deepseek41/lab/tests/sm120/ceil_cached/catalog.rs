//! The one-pass cached ceil entries paired with the served parallel decode
//! quantizers they would replace.
use bundle_deepseek41::sm120::ceil_parallel as registration;
use kernel_build::CubinSpec;
use kernels_sm120::row::block_quantize::{self, Entry};
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::GEOMETRIES};
use std::collections::BTreeMap;

pub const FAMILY: &str = "ceil_cached";
pub const MODULE: &str = "ceil_cached_sm120";

/// Every served parallel ceil quantizer of at most 8192 columns (eight
/// groups per warp) once, each beside its cached twin. Wider rows keep the
/// parallel entry.
pub fn pairs() -> Vec<[Entry; 2]> {
    let mut pairs = BTreeMap::new();
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            let Ok(served) = registration::quantizers(g, ranks) else {
                continue;
            };
            for old in served.into_iter().filter(|e| e.query.width <= 8192) {
                let mut new = old.clone();
                new.symbol = format!(
                    "{}_cached_v1",
                    old.symbol.strip_suffix("_parallel_v1").unwrap()
                );
                let new = new.cached().unwrap();
                pairs.insert(old.symbol.clone(), [old, new]);
            }
        }
    }
    pairs.into_values().collect()
}
pub fn specs() -> Vec<CubinSpec> {
    let entries: Vec<_> = pairs().into_iter().map(|[_, new]| new).collect();
    vec![block_quantize::cubin_spec(MODULE, &entries).unwrap()]
}
