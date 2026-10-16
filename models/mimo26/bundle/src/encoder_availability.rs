//! Encoder admission over cubin membership.
use model_mimo26_contract::backend::{Kernels, Selected};
use model_mimo26_contract::topology::Tp;

/// Exact geometry and topology membership of an embedded cubin.
pub fn cubin_serves(name: &str, tp: Tp) -> bool {
    crate::GEOMETRY_CUBINS
        .iter()
        .any(|(cubin, geometry, ranks)| {
            *cubin == name
                && (*geometry == "*" || *geometry == tp.geometry.id)
                && (*ranks == 0 || *ranks == tp.ranks())
        })
}

/// Encoder use requires every declared entry in a cubin serving the topology.
pub fn encoders_available(tp: Tp) -> bool {
    if !tp.geometry.encoders {
        return false;
    }
    let contains = |symbol: &str| {
        crate::CUBINS
            .iter()
            .any(|cubin| cubin_serves(cubin.name, tp) && cubin.symbols.contains(&symbol))
    };
    let Ok(symbols) = Selected::encoder_symbols(tp.geometry) else {
        return false;
    };
    symbols.iter().all(|symbol| contains(symbol))
}
