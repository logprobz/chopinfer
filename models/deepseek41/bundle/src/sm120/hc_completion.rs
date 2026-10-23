//! One-launch decode HC boundaries replacing the served chain producer and
//! fused finish/norm tail: the tail layout and the split norm layout.
use kernel_build::CubinSpec;
use kernels_sm120::sinkhorn_hc::completion::{self, Entry, Layout, Mutation};
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "sinkhorn_hc_completion";
pub const CUBIN: &str = "sinkhorn_hc_deepseek41_completion_sm120";
pub fn entry(layout: Layout) -> Entry {
    Entry {
        reference: super::hc_finish_norm::entry(&FLASH),
        producer_prefix: super::hc_round_chain::entry(&FLASH).prefix,
        prefix: format!("chop_{}_hc_completion", FLASH.kernel_tag),
        layout,
        mutation: Mutation::None,
    }
}
/// The layout served at C1 (sixteen warps per chain CTA) and C8 (eight)
/// until the column-block pair (`hc_cols::BUCKETS`) replaced it there; the
/// labs compare against it.
pub fn served() -> Entry {
    entry(Layout::Split)
}
/// Lab controls: each must differ from the served pair on real rows.
pub fn controls() -> Vec<Entry> {
    [
        (Layout::Tail, Mutation::ClearStatus),
        (Layout::Tail, Mutation::WrongFloor),
        (Layout::Split, Mutation::ClearStatus),
        (Layout::Split, Mutation::WrongFloor),
        (Layout::Split, Mutation::ReverseFold),
    ]
    .into_iter()
    .map(|(layout, mutation)| {
        let mut e = entry(layout);
        e.prefix = format!(
            "chop_hc_completion_{}",
            format!("{mutation:?}").to_lowercase()
        );
        e.mutation = mutation;
        e
    })
    .collect()
}
/// Every candidate warp count of both layouts, then each control at eight
/// warps.
pub fn entries() -> Vec<(Entry, u32)> {
    [Layout::Tail, Layout::Split]
        .into_iter()
        .flat_map(|layout| layout.warps().iter().map(move |&w| (entry(layout), w)))
        .chain(controls().into_iter().map(|e| (e, 8)))
        .collect()
}
pub fn cubin() -> Result<CubinSpec, String> {
    completion::cubin_spec(CUBIN, &entries()).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn symbols_plans_and_arguments_cover_both_layouts() {
        let spec = cubin().unwrap();
        assert_eq!(spec.entries.len(), 10);
        let mut symbols = std::collections::BTreeSet::new();
        for e in &spec.entries {
            assert!(symbols.insert(e.symbol.clone()), "{}", e.symbol);
            assert_eq!(e.params.len(), completion::ARGUMENTS);
        }
        for layout in [Layout::Tail, Layout::Split] {
            let e = entry(layout);
            for rows in [1, 8, 32] {
                for &w in layout.warps() {
                    let p = e.plan(rows, rows, w).unwrap();
                    let chains = 160 + u32::from(layout == Layout::Split);
                    assert_eq!((p.grid, p.block), ([chains, rows, 1], [w * 32, 1, 1]));
                    assert_eq!(p.workspace_bytes, u64::from(rows) * 4096 * 4);
                    assert!(p.pdl);
                    assert!(e.plan(rows, rows + 1, w).is_err());
                }
            }
            assert!(e.plan(1, 1, 4).is_err());
        }
        assert!(entry(Layout::Split).plan(1, 1, 32).is_err());
        assert_eq!(
            CUBIN,
            format!("sinkhorn_hc_{}_completion_sm120", FLASH.kernel_tag)
        );
        let mut bad = entry(Layout::Tail);
        bad.mutation = Mutation::ReverseFold;
        assert!(bad.validate().is_err());
    }
}
