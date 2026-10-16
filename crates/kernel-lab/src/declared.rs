//! The declarations of the lab's entries of families no catalog lists,
//! each at its served shape (`kernels_sm120::invariance::Declared`): the
//! Sinkhorn mixer's, latent attention's, the pooled indexer's, the
//! per-channel-decay rule's and the chained MoE engine's entries (with its
//! routed experts split over the ranks), which
//! the invariance adapters test (`tests/common/invariance.rs`,
//! `inventory`).

use kernels_sm120::invariance::{Declared, OpKind};
use kernels_sm120::latent::{self, merge_columns, slices};
use kernels_sm120::linear::channel_decay;
use kernels_sm120::moe::chained::{self, RoutedSumOutput, formats, split as moe_split};
use kernels_sm120::pooled_indexer::{self, merge_radix, merge_tree, rows};
use kernels_sm120::sinkhorn_hc::{self, SERVED_PROFILE, mix_norm_tree, split, tiled};

/// Every declaration of the lab's family entries.
pub fn lab_families() -> Vec<Declared> {
    let mut all = sinkhorn();
    all.extend(latent());
    all.extend(pooled_indexer());
    all.extend(channel_decay());
    all.extend(chained_moe());
    all.extend(chained_moe_split());
    all
}

/// The chained MoE engine's routed experts split over the ranks
/// (`chained::split`), as served: every form of `split::forms`, every route
/// kernel of the routes over all experts, the route sum in both forms and
/// its F32 add form.
pub fn chained_moe_split() -> Vec<Declared> {
    let declared = |symbol: String, invariance| Declared {
        symbol,
        profile: None,
        op: OpKind::ChainedMoe,
        invariance,
    };
    let mut all = moe_split::forms()
        .into_iter()
        .map(|p| declared(p.symbol.clone(), p.invariance()))
        .collect::<Vec<_>>();
    for route in moe_split::routes() {
        for &kernel in route.kernels() {
            all.push(declared(
                route.symbol(moe_split::PREFIX, kernel),
                route.invariance(kernel),
            ));
        }
    }
    let [sum] = moe_split::sums();
    for output in [RoutedSumOutput::Bf16, RoutedSumOutput::F32] {
        let sum = sum.with_output(output);
        all.push(declared(sum.symbol(moe_split::PREFIX), sum.invariance()));
    }
    let f32_sum = sum.with_output(RoutedSumOutput::F32);
    all.push(declared(
        f32_sum.add_symbol(moe_split::PREFIX),
        f32_sum.invariance(),
    ));
    all
}

/// The chained MoE engine's served dense entries the lab builds
/// (`chained::TP4_288_TOP8`): the catalog's projections in the column,
/// block-warp and decoded-tile schedules and `chained::formats`' catalog.
/// `formats::lab_projections`, which nothing serves, are left out; the
/// routed experts' entries are [`chained_moe_split`]'s.
pub fn chained_moe() -> Vec<Declared> {
    use crate::families::moe_chained_decoded;
    let declared = |symbol: String, invariance| Declared {
        symbol,
        profile: None,
        op: OpKind::ChainedMoe,
        invariance,
    };
    chained::catalog()
        .into_iter()
        .chain(chained::block_warp_catalog())
        .chain(formats::catalog())
        .chain(moe_chained_decoded::shapes::projections().to_vec())
        .map(|p| declared(p.symbol.clone(), p.invariance()))
        .collect()
}

/// The per-channel-decay rule's entries at the served shape
/// (`channel_decay::SERVED`): the staged catalog (convolutions, delta
/// rule, gated norm) and the tiled and streamed prefill stages.
pub fn channel_decay() -> Vec<Declared> {
    channel_decay::catalog()
        .into_iter()
        .chain(channel_decay::tiled_catalog())
        .chain(channel_decay::streamed_catalog())
        .map(|entry| Declared {
            invariance: entry.invariance(),
            symbol: entry.symbol,
            profile: Some(entry.profile),
            op: OpKind::ChannelDecay,
        })
        .collect()
}

/// The pooled indexer's entries at the served shape
/// (`pooled_indexer_shapes.rs`): every entry's symbol and split merge, and
/// the one-split, tree-merge and radix-merge forms SM120 runs in a
/// selection's place.
pub fn pooled_indexer() -> Vec<Declared> {
    use crate::families::pooled_indexer::shapes;
    let entries = shapes::lab_entries();
    let declared = |(symbol, invariance)| Declared {
        symbol,
        profile: Some(shapes::PROFILE),
        op: OpKind::PooledIndexer,
        invariance,
    };
    let mut all = entries
        .iter()
        .flat_map(pooled_indexer::Entry::declared_symbols)
        .map(declared)
        .collect::<Vec<_>>();
    all.extend(
        rows::entries(&entries, 1)
            .iter()
            .map(|form| declared((form.symbol(), form.invariance()))),
    );
    all.extend(
        merge_tree::entries(&entries)
            .iter()
            .map(|form| declared((form.symbol(), form.invariance()))),
    );
    all.extend(
        merge_radix::entries(&entries)
            .iter()
            .map(|form| declared((form.symbol(), form.invariance()))),
    );
    all
}

/// The served latent attention entries at the lab's shape (the served
/// geometry, `latent_shapes`): every entry's symbol and split merge, and
/// the slice-widened and column-merge forms SM120 runs in their places.
pub fn latent() -> Vec<Declared> {
    let profile = crate::latent_shapes::PROFILE;
    let entries = latent::entries(profile, crate::latent_shapes::GEOMETRY);
    let declared = |(symbol, invariance)| Declared {
        symbol,
        profile: Some(profile),
        op: OpKind::LatentAttention,
        invariance,
    };
    let mut all = entries
        .iter()
        .flat_map(latent::Entry::declared_symbols)
        .map(declared)
        .collect::<Vec<_>>();
    all.extend(
        slices::entries(&entries)
            .iter()
            .map(|form| declared((form.symbol(), form.invariance()))),
    );
    all.extend(
        merge_columns::entries(&entries)
            .iter()
            .map(|form| declared((form.symbol(), form.invariance()))),
    );
    all
}

/// The served Sinkhorn mixer's entries: the one-CTA mix-collapse, expand
/// and head, the split mix-collapse, the served tiles and the split fused
/// with the tree norm.
pub fn sinkhorn() -> Vec<Declared> {
    let declared = |symbol: &str, invariance| Declared {
        symbol: symbol.to_owned(),
        profile: Some(SERVED_PROFILE),
        op: OpKind::Sinkhorn,
        invariance,
    };
    let mut all = sinkhorn_hc::catalog()
        .iter()
        .map(|entry| declared(&entry.symbol, entry.invariance()))
        .collect::<Vec<_>>();
    let split = split::served();
    all.push(declared(&split.symbol, split.invariance()));
    for tile in tiled::SERVED_TILES {
        let entry = tiled::served(tile);
        all.push(declared(&entry.symbol, entry.invariance()));
    }
    let fused = mix_norm_tree::served();
    all.push(declared(&fused.symbol, fused.invariance()));
    all
}
