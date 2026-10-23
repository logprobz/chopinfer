//! The column-block HC producer and its split finish for DeepSeek's wide
//! decode buckets: one CTA per 128-column block of the four streams and a
//! tile of rows produces the boundary's expansion and chain sums, then the
//! two-CTA finish sums the RMS lanes from the expanded streams before the
//! served fold, Sinkhorn, collapse and norm. A matched pair: both launches
//! replace the served two-launch boundary together, over its arguments.
use kernel_build::CubinSpec;
use kernels_sm120::sinkhorn_hc::{finish_split, round_cols};
use model_deepseek41_contract::geometry::FLASH;
pub const FAMILY: &str = "sinkhorn_hc_cols";
/// Measured decode buckets of the pair: four rows per producer CTA with the
/// transposed butterfly and a 32-warp finish at 16 and 32 rows. On 385
/// boundaries of real captured rows (TP4 GPUs at 300 W, 2026-10-22) every
/// output, chain sum and control matched the served pair bit for bit; per
/// boundary, cold, it took 18.0 us against 32.0 at 16 rows and 19.2 to 20.1
/// us against 36.5 to 38.2 at 32 rows, the fastest of the 18 forms at both.
pub const BUCKETS: &[finish_split::Bucket] = &[
    finish_split::Bucket {
        rows: 16,
        tile: 4,
        transposed: true,
        warps: 32,
    },
    finish_split::Bucket {
        rows: 32,
        tile: 4,
        transposed: true,
        warps: 32,
    },
];
pub const PRODUCER_CUBIN: &str = "sinkhorn_hc_deepseek41_round_cols_sm120";
pub const FINISH_CUBIN: &str = "sinkhorn_hc_deepseek41_finish_split_sm120";

pub fn producer() -> round_cols::Entry {
    round_cols::Entry {
        reference: super::hc_round_chain::entry(&FLASH),
        prefix: format!("chop_{}_hc_round_cols", FLASH.kernel_tag),
        mutation: round_cols::Mutation::None,
    }
}
pub fn finish() -> finish_split::Entry {
    finish_split::Entry {
        reference: super::hc_finish_norm::entry(&FLASH),
        prefix: format!("chop_{}_hc_finish_split", FLASH.kernel_tag),
        mutation: finish_split::Mutation::None,
    }
}
/// Lab controls: each must differ from the served boundary on real rows.
pub fn producer_controls() -> Vec<round_cols::Entry> {
    vec![round_cols::Entry {
        prefix: "chop_hc_round_cols_reverse_levels".into(),
        mutation: round_cols::Mutation::ReverseLevels,
        ..producer()
    }]
}
pub fn finish_controls() -> Vec<finish_split::Entry> {
    [
        finish_split::Mutation::ClearStatus,
        finish_split::Mutation::WrongFloor,
        finish_split::Mutation::ReverseFold,
        finish_split::Mutation::ReverseSquares,
    ]
    .into_iter()
    .map(|mutation| finish_split::Entry {
        prefix: format!(
            "chop_hc_finish_split_{}",
            format!("{mutation:?}").to_lowercase()
        ),
        mutation,
        ..finish()
    })
    .collect()
}
/// Every candidate tile, then each control at four rows per CTA.
pub fn producer_entries() -> Vec<(round_cols::Entry, u32, bool)> {
    round_cols::TILES
        .iter()
        .map(|&(tile, transposed)| (producer(), tile, transposed))
        .chain(producer_controls().into_iter().map(|e| (e, 4, false)))
        .collect()
}
/// Every candidate warp count, then each control at 32 warps.
pub fn finish_entries() -> Vec<(finish_split::Entry, u32)> {
    finish_split::WARPS
        .iter()
        .map(|&w| (finish(), w))
        .chain(finish_controls().into_iter().map(|e| (e, 32)))
        .collect()
}
pub fn cubins() -> Result<[CubinSpec; 2], String> {
    Ok([
        round_cols::cubin_spec(PRODUCER_CUBIN, &producer_entries()).map_err(|e| e.to_string())?,
        finish_split::cubin_spec(FINISH_CUBIN, &finish_entries()).map_err(|e| e.to_string())?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{
        backend::Launch,
        contract::{Argument, ArgumentList},
    };
    #[test]
    fn symbols_plans_and_arguments_keep_the_served_boundary() {
        let [producers, finishes] = cubins().unwrap();
        assert_eq!(producers.entries.len(), 7);
        assert_eq!(finishes.entries.len(), 7);
        let chain = super::super::hc_round_chain::entry(&FLASH);
        let reference = chain.cubin_spec("chain").unwrap().entries.remove(0);
        let tail = super::super::hc_finish_norm::entry(&FLASH);
        let tail_params = tail.cubin_spec("tail").unwrap().entries.remove(1).params;
        let mut symbols = std::collections::BTreeSet::new();
        for e in &producers.entries {
            assert!(symbols.insert(e.symbol.clone()), "{}", e.symbol);
            assert_eq!(e.params, reference.params);
        }
        for e in &finishes.entries {
            assert!(symbols.insert(e.symbol.clone()), "{}", e.symbol);
            assert_eq!(e.params, tail_params);
        }
        for rows in [16, 32] {
            let served = chain.plan(rows, rows, 4).unwrap();
            for (tile, tx) in round_cols::TILES {
                let p = producer().plan(rows, rows, tile, tx).unwrap();
                assert_eq!(
                    (p.grid, p.block),
                    ([40, rows.div_ceil(tile), 1], [256, 1, 1])
                );
                assert_eq!(p.workspace_bytes, served.workspace_bytes);
                assert!(p.pdl);
                let launch = Launch {
                    symbol: chain.symbols()[0].clone(),
                    plan: served,
                    rows,
                    arguments: ArgumentList((0..9).map(Argument::DeviceBuffer).collect()),
                    collective: false,
                    persistent: false,
                };
                let new = producer().compose(&launch, tile, tx).unwrap();
                assert_eq!(new.arguments, launch.arguments);
                assert_eq!(new.plan, p);
                let mut other = launch.clone();
                other.arguments.0.pop();
                assert!(producer().compose(&other, tile, tx).is_err());
            }
            for w in finish_split::WARPS {
                let p = finish().plan(rows, rows, w).unwrap();
                let served = tail.plan(rows, rows, 32).unwrap();
                assert_eq!((p.grid, p.block), ([2, rows, 1], [w * 32, 1, 1]));
                assert_eq!(p.workspace_bytes, served.workspace_bytes);
                assert_eq!(p.dynamic_shared_bytes, served.dynamic_shared_bytes);
                let launch = Launch {
                    symbol: tail.symbols()[1].clone(),
                    plan: served,
                    rows,
                    arguments: ArgumentList((0..18).map(Argument::DeviceBuffer).collect()),
                    collective: false,
                    persistent: false,
                };
                let new = finish().compose(&launch, w).unwrap();
                assert_eq!(new.arguments, launch.arguments);
                let mut other = launch.clone();
                other.symbol = tail.symbols()[0].clone();
                assert!(finish().compose(&other, w).is_err());
            }
        }
        assert_eq!(
            PRODUCER_CUBIN,
            format!("sinkhorn_hc_{}_round_cols_sm120", FLASH.kernel_tag)
        );
        assert_eq!(
            FINISH_CUBIN,
            format!("sinkhorn_hc_{}_finish_split_sm120", FLASH.kernel_tag)
        );
    }
}
