//! MiMo's SM120 selection: the `kernels_sm120` backend over MiMo's own
//! catalog entries. Model code reaches kernels through
//! `kernel_core::backend` (`crate::ops`); only this module names
//! `kernels_sm120`.

pub mod attention;
pub mod comm;
#[cfg(test)]
pub mod comm_plan;
pub mod dense;
#[cfg(test)]
pub mod device;
pub mod encoders;
pub mod launch;
pub mod modules;
pub mod overlap;
pub mod own_ops;
pub mod projection;
#[cfg(test)]
pub mod reference;
pub mod ring_entries;
pub mod selection;
pub mod target;

use kernel_core::backend::Backend;
use kernel_core::contract::{DType, Refusal};
use kernels_sm120::backend::{RopeKvEntry, Sm120};
use kernels_sm120::row::RowInstantiation;

use crate::program::Stage;
use crate::selection::{Phase, RowBuckets};
use crate::switches::Switches;
use crate::topology::Tp;
use selection::{Selected, select};

/// The SM120 backend of a `tp` instance with `switches`: the entries
/// MiMo's catalog instantiates for the ops `crate::ops` prepares. Graphs
/// take the wide all-reduce entries only when `wide`
/// (`switches::WIDE_COMM` and the residency check).
pub fn backend(tp: Tp, switches: Switches, wide: bool) -> Result<Box<dyn Backend>, Refusal> {
    let (plain, combine, invariance) = comm::exact_entries(tp)?;
    let backend = Sm120::new()
        .with_rmsnorm(
            crate::ops::hidden_norm(tp),
            selection::entry_symbol(tp, "row", "rms_hidden"),
        )
        .with_exact_all_reduce(plain, combine, invariance)
        .with_rope_kv(rope_entries(tp, switches)?)
        .with_attention(attention::entries(tp, switches))
        .with_row_copy(row_copy());
    let backend = row_entries(tp)?
        .into_iter()
        .fold(backend, |backend, entry| backend.with_row(entry));
    let backend = head_norm_entries(tp)?
        .into_iter()
        .fold(backend, |backend, entry| backend.with_row_rmsnorm(entry));
    let backend = with_moe_routes(backend, tp)?;
    Ok(Box::new(if wide {
        backend
    } else {
        backend.without_wide_collectives()
    }))
}

/// The shared column slice entry (`row::copy`), which MTP's EH shards
/// launch; the build compiles it into `row_copy_sm120`.
pub fn row_copy() -> kernels_sm120::row::copy::Entry {
    use kernels_sm120::row::copy::{Entry, Pattern, SLICE_SYMBOL};
    Entry::new(Pattern::ColumnSlice, SLICE_SYMBOL)
}

/// The row engine entries of MiMo's embedding and head-row gathers and of
/// the drafters' sharded argmax.
fn row_entries(tp: Tp) -> Result<Vec<RowInstantiation>, Refusal> {
    let mut entries = vec![match selection::row_for(tp, "embedding_full")? {
        Selected::Row(entry) => entry,
        _ => return Err(Refusal("MiMo's embedding is a row entry")),
    }];
    for name in ["dflash_argmax_partial", "dflash_argmax_final"] {
        let symbol = selection::drafter_row_symbol(tp, name);
        entries.push(
            crate::sm120::launch::row::entries_for(tp)
                .find(|entry| entry.symbol == symbol)
                .cloned()
                .ok_or(Refusal("MiMo sharded argmax entry missing"))?,
        );
    }
    Ok(entries)
}

/// MiMo's MoE routes and combine: the row router with the 64-row tiled
/// permutation (prefill and buckets that route as prefill does), the
/// sigmoid-bias decode route of 16 tokens (which declares nothing), the
/// fused router and route of the smaller buckets, and the combine of the
/// routed rows.
fn with_moe_routes(backend: Sm120, tp: Tp) -> Result<Sm120, Refusal> {
    let row = |name: &str| match selection::row_for(tp, name)? {
        Selected::Row(entry) => Ok(entry),
        _ => Err(Refusal("MiMo's MoE rows are row entries")),
    };
    let fused = model_mimo26_contract::sm120::moe::fused_router_entries(&crate::topology::tagged(
        model_mimo26_contract::sm120::moe::FUSED_ROUTER_PREFIX,
        tp,
    ))?;
    let backend = fused
        .into_iter()
        .fold(backend, |backend, entry| backend.with_fused_router(entry));
    Ok(backend
        .with_router(row("router")?)
        .with_tile_permute([
            row("permute_tile_count_mimo_b64")?,
            row("permute_tile_prefix_mimo_b64")?,
            row("permute_tile_scan_mimo_b64")?,
            row("permute_tile_scatter_mimo_b64")?,
        ])
        .with_sigmoid_decode_route(
            crate::topology::tagged(model_mimo26_contract::sm120::moe::DECODE_ROUTE_SYMBOL, tp),
            model_mimo26_contract::serving::DECODE_EXPERT_ROWS,
            kernels_sm120::invariance::Invariance::NONE,
        )
        .with_expert_combine(row("combine_hidden")?))
}

/// DFlash's per-head RMSNorm entries of its Q and K heads.
fn head_norm_entries(tp: Tp) -> Result<Vec<RowInstantiation>, Refusal> {
    ["dflash_q_norm", "dflash_k_norm"]
        .map(|name| {
            let symbol = selection::drafter_row_symbol(tp, name);
            crate::sm120::launch::row::entries_for(tp)
                .find(|entry| entry.symbol == symbol)
                .cloned()
                .ok_or(Refusal("MiMo DFlash head norm entry missing"))
        })
        .into_iter()
        .collect()
}

/// The backend of a registered lookup launcher (`crate::projection`): the
/// embedding entry the precision component authenticated for it.
pub fn lookup_backend(entry: RowInstantiation) -> Box<dyn Backend> {
    Box::new(Sm120::new().with_row(entry))
}

/// The RoPE, KV write and ring write entries MiMo's steps select: the
/// paged writes of both page formats, the sliding-window RoPE of decode
/// and of prefill and verify, the ring write, and DFlash's Q and K RoPE.
/// Ring entries take the block table width the paged ones are built for.
fn rope_entries(tp: Tp, switches: Switches) -> Result<Vec<RopeKvEntry>, Refusal> {
    let g = tp.geometry;
    let global = (0..g.layers()).find(|&layer| g.is_global(layer));
    let sliding = (0..g.layers()).find(|&layer| !g.is_global(layer));
    let (Some(global), Some(sliding)) = (global, sliding) else {
        return Err(Refusal("MiMo has global and sliding-window layers"));
    };
    let row = |selected| match selected {
        Selected::Row(entry) => Ok(entry),
        _ => Err(Refusal("MiMo RoPE stages select row entries")),
    };
    let mut entries = Vec::new();
    for kv in [DType::Bf16, DType::Fp8E4M3] {
        entries.push(row(select(
            Stage::RopeAndPagedWrite(global),
            Phase::Prefill,
            kv,
            tp,
            switches,
        )?)?);
    }
    for phase in [Phase::Prefill, Phase::Decode(1)] {
        entries.push(row(select(
            Stage::RopeForSlidingWindow(sliding),
            phase,
            DType::Bf16,
            tp,
            switches,
        )?)?);
    }
    entries.push(row(select(
        Stage::SlidingWindowWrite(sliding),
        Phase::Prefill,
        DType::Bf16,
        tp,
        switches,
    )?)?);
    for name in ["dflash_q_rope_table_pos32", "dflash_k_rope_table_pos32"] {
        if let Ok(selected) = selection::row_for(tp, name) {
            entries.push(row(selected)?);
        }
    }
    let max_pages = tp.model_limits().max_pages();
    Ok(entries
        .into_iter()
        .map(|entry| RopeKvEntry::new(entry).with_ring_max_pages(max_pages))
        .collect())
}

/// The declarations of the bundle's own kernels that prefill and exact
/// verify steps launch (`kernels_sm120::invariance`): the ring backup copies
/// each slot's window ring as the phase starts, without arithmetic, at
/// every slot count an instance may hold (one per item of its largest
/// bucket, [`ring_entries::ring_symbol`]).
pub fn declared_kernels() -> Vec<kernels_sm120::invariance::Declared> {
    use kernels_sm120::invariance::{Declared, Invariance, OpKind};
    let rings = bundle_runtime::geometry::Items::package::<crate::contract::Mimo>();
    crate::topology::all()
        .flat_map(|tp| {
            rings
                .buckets()
                .map(move |slots| ring_entries::ring_symbol(tp, "ring_backup", slots))
        })
        .map(|symbol| Declared {
            symbol,
            profile: None,
            op: OpKind::Bundle,
            invariance: Invariance::COPY,
        })
        .collect()
}

/// The row buckets of the SM120 row kernels, which eager prefill row
/// kernels grid (`MimoPhase::row_grid`).
pub const ROW_BUCKETS: RowBuckets = RowBuckets::new(kernels_sm120::row::smallest_bucket);

/// Whether a decode or verify graph of `rows` rows folds the MoE combine
/// into its MLP all-reduce (`comm::graph_folds_combine_for`).
pub fn folds_combine(tp: Tp, rows: u32) -> bool {
    comm::graph_folds_combine_for(tp, rows)
}
