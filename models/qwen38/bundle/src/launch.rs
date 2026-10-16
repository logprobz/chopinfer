//! The names of the bundle's entries as an instance's geometry tags them,
//! and the launch switches and row counts its programs share.

use crate::contract::{DECODE_ENTRY_ROWS, Shape};
use crate::scratch::Served;

/// Rewrite the source geometry tag at its symbol token, preserving symbols
/// already emitted for the target geometry or its ranked variant.
fn retag(symbol: &str, target: &str) -> String {
    let source = model_qwen38_contract::geometry::FLASH.kernel_tag;
    let Some((offset, _)) = symbol.match_indices(source).find(|(i, _)| {
        (*i == 0 || symbol.as_bytes()[*i - 1] == b'_')
            && symbol
                .as_bytes()
                .get(*i + source.len())
                .is_none_or(|c| *c == b'_')
    }) else {
        return symbol.to_owned();
    };
    let tail = &symbol[offset..];
    if tail
        .strip_prefix(target)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
    {
        return symbol.to_owned();
    }
    format!(
        "{}{target}{}",
        &symbol[..offset],
        &symbol[offset + source.len()..]
    )
}

/// A rank-invariant symbol for the instance geometry.
pub fn tagged(symbol: &str, shape: Shape) -> String {
    retag(symbol, shape.geometry.kernel_tag)
}

/// A per-rank symbol for the instance geometry.
pub fn per_rank(symbol: &str, shape: Shape) -> String {
    retag(symbol, shape.tag())
}

/// Environment switch: `0` keeps decode graphs above 32 rows on the phased
/// all-reduce pair instead of the wide entry (an A/B of the wide path).
pub const WIDE_COMM: &str = "CHOPINFER_QWEN38_WIDE_COMM";

/// Whether decode graphs above 32 rows may take the wide all-reduce: unless
/// `WIDE_COMM` is `0`.
pub fn wide_comm_enabled(value: Option<&str>) -> bool {
    value != Some("0")
}

/// Items one prefill phase may carry: every served sequence, and never
/// fewer than the 32 an instance was sized for before the served count, so
/// 32 or fewer served items keep that workspace.
pub const fn prefill_sequences(served: Served) -> u32 {
    if served.items > DECODE_ENTRY_ROWS {
        served.items
    } else {
        DECODE_ENTRY_ROWS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_tags_rewrite_only_the_model_token() {
        let source = model_qwen38_contract::geometry::FLASH.kernel_tag;
        let symbol = format!("chop_row_{source}_norm_v1");
        for tag in ["small", "row", "qwen3", "gemm", "qwen38_synthetic_small"] {
            let rewritten = retag(&symbol, tag);
            assert_eq!(rewritten, format!("chop_row_{tag}_norm_v1"));
            assert_eq!(retag(&rewritten, tag), rewritten);
        }
    }

    #[test]
    fn the_wide_all_reduce_is_on_unless_switched_off() {
        assert!(wide_comm_enabled(None));
        assert!(wide_comm_enabled(Some("1")));
        assert!(!wide_comm_enabled(Some("0")));
    }
}
