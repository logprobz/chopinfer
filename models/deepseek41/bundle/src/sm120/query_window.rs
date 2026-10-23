//! The one-row decode query RoPE and fused window write of Flash TP4 in one
//! clustered launch (`kernels_sm120::latent::query_window_twin`): the
//! served RoPE (`latent_scheduled::query`) and the served fused window
//! write (`window_fused`) of the layer, with the window's first status
//! commit after the RoPE's, as served. An instance loads and registers the
//! twins only where they serve ([`serves`]).
use kernel_build::CubinSpec;
use kernels_sm120::latent::query_window_twin::{self, Entry};
use model_deepseek41_contract::geometry::{FLASH, Geometry};

pub const FAMILY: &str = "query_window_twin";
pub const CUBIN: &str = "query_window_twin_deepseek41_sm120";

/// The rows a twin holds: the one-row decode step, whose KV projection the
/// dual fold writes before the query chain.
pub const ROWS: u32 = 1;

/// Whether the twins serve: the lab pair (d2-0463g3h) kept every output
/// bit and saved 2 us per layer.
pub const SERVED: bool = true;

/// Set to zero to keep the served RoPE and window write apart.
pub const SWITCH: &str = "CHOPINFER_DEEPSEEK41_QUERY_WINDOW_TWIN";

/// One twin per window format, beside the served fused window entry of the
/// same format.
pub fn entries() -> Vec<Entry> {
    super::window_fused::entries()
        .into_iter()
        .map(|window| Entry {
            kv: window.kv,
            capacity: ROWS,
            symbol: window
                .symbol
                .replace("_window_fused_", "_query_window_twin_"),
        })
        .collect()
}

/// Whether `g` at `ranks` runs the twins: Flash TP4 once they serve, unless
/// [`SWITCH`] is zero.
pub fn serves(g: &Geometry, ranks: u32) -> bool {
    if g.id != FLASH.id || ranks != 4 {
        return false;
    }
    #[cfg(test)]
    if let Some(forced) = FORCED.get() {
        return forced;
    }
    SERVED && std::env::var(SWITCH).as_deref() != Ok("0")
}
#[cfg(test)]
std::thread_local! {
    /// Tests keep the served RoPE and window write apart unless they serve
    /// the twins, so their pinned journals stay the pair's.
    static FORCED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(Some(false)) };
}
/// Run `run` with the twins served (`true`) or kept apart (`false`),
/// whatever [`SERVED`] and [`SWITCH`] say.
#[cfg(test)]
pub(crate) fn with_served<T>(served: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCED.set(self.0);
        }
    }
    let _restore = Restore(FORCED.replace(Some(served)));
    run()
}

pub fn cubins(geometries: &[Geometry]) -> Result<Vec<(&'static str, CubinSpec)>, String> {
    if !geometries.iter().any(|g| g.id == FLASH.id) {
        return Ok(Vec::new());
    }
    Ok(vec![(
        FAMILY,
        query_window_twin::cubin_spec(CUBIN, &entries()).map_err(|e| e.to_string())?,
    )])
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::latent_attention::LatentKvFormat;

    #[test]
    fn one_twin_per_window_format_keeps_the_served_abis() {
        let twins = entries();
        assert_eq!(twins.len(), 2);
        let rope = super::super::latent_scheduled::query();
        for twin in &twins {
            let window = super::super::window_fused::entries()
                .into_iter()
                .find(|w| w.kv == twin.kv)
                .unwrap();
            assert_eq!(twin.window().spec().params, window.spec().params);
            assert_eq!(twin.rope().spec().params, rope.spec().params);
            assert_eq!(twin.capacity, ROWS);
            assert_eq!(twin.plan(1, 16).unwrap().cluster, [1, 2, 1]);
        }
        let symbols: Vec<_> = twins.iter().map(|t| t.symbol.as_str()).collect();
        assert_eq!(
            symbols,
            [
                "chop_deepseek41_query_window_twin_native_v1",
                "chop_deepseek41_query_window_twin_bf16_v1"
            ]
        );
        assert!(twins.iter().any(|t| t.kv == LatentKvFormat::Bf16));
        assert_eq!(
            cubins(model_deepseek41_contract::geometry::GEOMETRIES)
                .unwrap()
                .len(),
            1
        );
        assert!(!serves(&FLASH, 4));
    }
}
