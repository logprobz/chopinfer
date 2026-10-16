//! The bundle's context-parallel entries at TP4, group size 2 (the two ranks
//! of a KV head's replica set): the owner-routed K/V and indexer key writes
//! (`kernels_sm120::owned_writes`), the head-sum per-rank selection
//! (`pooled_indexer::head_sum`) and its exact merge into block ids
//! (`pooled_indexer::block_merge`), the selected attention partials over a
//! rank's own pages (`selected_partials`), and the partial merge and the
//! position localization (`context_parallel`). The group all-gather is the
//! shared comm engine's (`context_parallel::group::ALLGATHER_SYMBOL`), which
//! the served `comm_sm120` cubin already holds. Every cubin here gates on its
//! own lock family, [`FAMILY`], so the served cubins and `kernels.lock` stay
//! as they are, and an instance loads them only at a size above 1.

use kernel_build::CubinSpec;
use kernel_core::contract::AttentionPhase;
use kernels_sm120::Profile;
use kernels_sm120::context_parallel::{self, GroupEntries, Kind, group};
use kernels_sm120::invariance::{Anchors, Chunk, Declared, Invariance, OpKind, Rows};
use kernels_sm120::owned_writes;
use kernels_sm120::pooled_indexer::{block_merge, head_sum};
use kernels_sm120::selected_partials;

use crate::contract::{PAGE_TOKENS, PREFILL_TOKENS, Ranks, Shape};
use crate::sm120::attention::{gated_config, sparse_geometry};
use crate::sm120::config::EXACT_VERIFY_ROWS;

/// The lock family of every context-parallel cubin.
pub const FAMILY: &str = "context_parallel";

/// The group size of every entry.
pub const RANKS: u32 = 2;

/// The group all-gather of the queries, the per-rank selections, each
/// owner's partials (targeted) and the barrier.
pub const ALLGATHER_SYMBOL: &str = group::ALLGATHER_SYMBOL;

// The group sizes, buckets and buffer sizes, at their earlier path.
pub use crate::context_parallel::{
    BARRIER_WORDS, GEOMETRIES, GroupDims, MAX_BUCKET, RankBytes, SIZES, bucket, bucket_index,
    buckets, dims, rank_bytes, sizes,
};

/// Whether the SM120 engines instantiate every context-parallel entry at
/// `shape`'s geometry. The head-sum selection, for one, takes four
/// indexer heads of 128 only.
pub fn has_entries(shape: Shape) -> bool {
    cubin_specs(shape).is_ok()
}

/// The TP4 profile whose per-rank shape the entries take.
fn profile(shape: Shape) -> Profile {
    Shape::new(shape.geometry, Ranks::TP4).profile()
}

/// The head-sum selection geometry: the replicated indexer's heads over
/// its blocks, keeping `top_blocks` blocks with the always-selected tail.
pub fn selection(shape: Shape) -> head_sum::Geometry {
    let g = sparse_geometry(profile(shape)).expect("a Qwen3.8 profile");
    head_sum::Geometry {
        heads: g.index_heads,
        head_dim: g.index_dim,
        group: g.block,
        page_size: g.page_size,
        max_top_k: g.top_blocks * g.block,
        always_tail: true,
    }
}

/// The selected partials' geometry: the rank's Q heads over its KV head.
pub fn partials_geometry(shape: Shape) -> selected_partials::Geometry {
    let g = sparse_geometry(profile(shape)).expect("a Qwen3.8 profile");
    selected_partials::Geometry {
        q_heads: g.q_heads,
        kv_heads: g.kv_heads,
        head_dim: g.head_dim,
        page: g.page_size,
        block: g.block,
        top_blocks: g.top_blocks,
    }
}

/// The merge and localization geometry: the rank's Q heads over the head
/// width, positions of 64-token pages, and the prefill capacity.
pub fn merge_geometry(shape: Shape) -> context_parallel::Geometry {
    let g = partials_geometry(shape);
    context_parallel::Geometry {
        heads: g.q_heads * g.kv_heads,
        width: g.head_dim,
        page_size: PAGE_TOKENS as u32,
        list_width: 1,
        capacity: PREFILL_TOKENS,
    }
}

/// The owner-routed writes: `kv_write_{bf16,fp8}` and `index_key_write`.
pub fn owned_entries(shape: Shape) -> Vec<owned_writes::Entry> {
    let profile = profile(shape);
    owned_writes::entries(
        profile,
        gated_config(profile).expect("a Qwen3.8 profile"),
        sparse_geometry(profile).expect("a Qwen3.8 profile"),
        RANKS,
    )
}

/// The per-rank selection of group size 2 and its split merge.
pub fn select_entry(shape: Shape) -> head_sum::Entry {
    head_sum::entries(profile(shape), selection(shape))
        .into_iter()
        .find(|entry| entry.kind == head_sum::Kind::Select { ranks: Some(RANKS) })
        .expect("a per-rank head-sum selection")
}

/// The exact merge of both ranks' selections into block ids.
pub fn merge_entry(shape: Shape) -> block_merge::Entry {
    block_merge::entries(profile(shape), selection(shape), &[RANKS])
        .pop()
        .expect("a block-id merge")
}

/// The selected partials: BF16 and FP8 K/V, decode, verify and prefill.
pub fn partial_entries(shape: Shape) -> Vec<selected_partials::Entry> {
    selected_partials::entries(profile(shape), partials_geometry(shape))
}

/// The partial merge and the position localization of group size 2.
pub fn group_entries(shape: Shape) -> Vec<context_parallel::Entry> {
    context_parallel::entries(profile(shape), merge_geometry(shape))
        .into_iter()
        .filter(|entry| {
            entry.name.ends_with("_partial_merge2") || entry.name.ends_with("_localize_positions2")
        })
        .collect()
}

/// The position localization entry.
pub fn localize_entry(shape: Shape) -> context_parallel::Entry {
    group_entries(shape)
        .into_iter()
        .find(|entry| entry.kind == Kind::LocalizeKeys { ranks: RANKS })
        .expect("a position localization")
}

/// The partial merge entry.
pub fn partial_merge_entry(shape: Shape) -> context_parallel::Entry {
    group_entries(shape)
        .into_iter()
        .find(|entry| entry.kind == Kind::PartialMerge { ranks: RANKS })
        .expect("a partial merge")
}

/// Every cubin of the family, named after the TP4 profile: the owned
/// writes, the head-sum selection, the block merge, the selected partials
/// and the merge and localization.
pub fn cubin_specs(shape: Shape) -> Result<Vec<CubinSpec>, String> {
    let profile = profile(shape);
    fn refusal(what: &'static str) -> impl Fn(kernel_core::contract::Refusal) -> String {
        move |r| format!("{what}: {r}")
    }
    Ok(vec![
        owned_writes::cubin_spec(profile, &owned_entries(shape))
            .map_err(refusal("owner-routed writes"))?,
        head_sum::cubin_spec(profile, &[select_entry(shape)])
            .map_err(refusal("head-sum selection"))?,
        block_merge::cubin_spec(profile, &[merge_entry(shape)]).map_err(refusal("block merge"))?,
        selected_partials::cubin_spec(profile, &partial_entries(shape))
            .map_err(refusal("selected partials"))?,
        context_parallel::cubin_spec(profile, &group_entries(shape))
            .map_err(refusal("partial merge"))?,
    ])
}

/// The catalog's context-parallel groups (`ModelCatalog::context_parallel`):
/// the entries and cubins of every geometry's TP4 shape that serves group
/// size [`RANKS`].
pub fn groups() -> Vec<GroupEntries> {
    serving_shapes()
        .map(|shape| GroupEntries {
            profile: profile(shape),
            ranks: RANKS,
            owned: owned_entries(shape),
            select: select_entry(shape),
            block_merge: merge_entry(shape),
            partials: partial_entries(shape),
            group: group_entries(shape),
            cubins: cubin_specs(shape).expect("a serving geometry has every entry"),
        })
        .collect()
}

/// Every geometry's TP4 shape that serves group size [`RANKS`].
fn serving_shapes() -> impl Iterator<Item = Shape> {
    crate::geometry::GEOMETRIES
        .iter()
        .map(|g| Shape::new(g, Ranks::TP4))
        .filter(|&shape| sizes(shape).contains(&RANKS))
}

/// The catalog's context-parallel declarations: [`declared`] at every
/// shape of [`groups`].
pub fn declarations() -> Vec<Declared> {
    serving_shapes().flat_map(declared).collect()
}

/// The names of [`cubin_specs`]: an instance loads them only at a
/// context-parallel size above 1.
pub fn cubin_names(shape: Shape) -> Vec<String> {
    let tag = profile(shape).tag();
    [
        "owned_writes",
        "pooled_indexer_head_sum",
        "block_merge",
        "selected_partials",
        "context_parallel",
    ]
    .map(|engine| format!("{engine}_{tag}_sm120"))
    .into()
}

/// Whether cubin `name` belongs to the family for any geometry.
pub fn is_cubin(name: &str) -> bool {
    crate::geometry::GEOMETRIES.iter().any(|g| {
        cubin_names(Shape::new(g, Ranks::TP4))
            .iter()
            .any(|cubin| cubin == name)
    })
}

/// Every entry symbol of the family, with each selection's split merge.
pub fn symbols(shape: Shape) -> Vec<String> {
    let select = select_entry(shape);
    owned_entries(shape)
        .iter()
        .map(owned_writes::Entry::symbol)
        .chain([select.symbol()])
        .chain(select.merge_symbol())
        .chain([merge_entry(shape).symbol()])
        .chain(
            partial_entries(shape)
                .iter()
                .map(selected_partials::Entry::symbol),
        )
        .chain(
            group_entries(shape)
                .iter()
                .map(context_parallel::Entry::symbol),
        )
        .collect()
}

/// The exactness of every entry of [`symbols`] but the head-sum selection
/// entry and the decode partials, declared by symbol on the TP4 profile
/// (`OpKind::ContextParallel`). Each declared entry computes a row, or a
/// row and head, in a CTA or warp of its own in a fixed order from that
/// row's inputs only, so no split, grid or other row changes its bits.
/// Rows claims cover the exact verify rows ([`EXACT_VERIFY_ROWS`]); no
/// batch claims are made.
pub fn declared(shape: Shape) -> Vec<Declared> {
    let one_row = Rows::OneRow {
        max: EXACT_VERIFY_ROWS,
    };
    // Every row in any chunk, and one-row bits at the verify rows.
    let rows = Invariance {
        chunk: Chunk::Any,
        rows: one_row,
        ..Invariance::NONE
    };
    let mut declared = Vec::new();
    for entry in owned_entries(shape) {
        let invariance = match entry.kind {
            // The served gated K/V write body unchanged, but for an early
            // return from the K/V warps of a row another rank owns.
            owned_writes::Kind::KvWrite { .. } => rows,
            // The served key write line for line, with only the block-key
            // row lookup replaced: a completed block key sums the block's
            // raw keys in double in position order, from the launch or from
            // the ring's BF16 copies of the same keys, so the bits hold at
            // block-aligned chunks. The spare anchor entry is copied from
            // the launch's rows alone (positions [end - ring, end)), so it
            // is complete only for an anchor at least a ring after the
            // launch's first row: the bundle's anchors, on multiples of its
            // 64-row prefill granule in launches that start on one, are.
            owned_writes::Kind::KeyWrite => Invariance {
                chunk: Chunk::Granule(entry.indexer.block),
                rows: one_row,
                anchors: Anchors::SpareAfter {
                    rows: entry.indexer.ring,
                },
                ..Invariance::NONE
            },
        };
        declared.push((entry.symbol(), invariance));
    }
    // The head-sum selection's own output is workspace: each split's kept
    // keys, whose bits follow the split count the prepared rows set, so
    // the entry declares nothing. Its split merge writes the composition's
    // output, the row's best-first pairs, and carries the composition's
    // claim (the selection op guarantees it): the selection scores each
    // pool by a fixed MMA chain on split ranges aligned from local pool 0,
    // and its streaming top-k and the split merge keep distinct (score,
    // pool) keys sorted, so the kept set and its order do not depend on
    // offer order or split count. The block-id merge folds both ranks'
    // pairs the same way, one CTA per row.
    declared.extend(
        select_entry(shape)
            .merge_symbol()
            .map(|symbol| (symbol, rows)),
    );
    declared.push((merge_entry(shape).symbol(), rows));
    for entry in partial_entries(shape) {
        // One CTA per (row, KV head) walks the row's owned keys in order
        // with no atomics, and the decode, verify and prefill entries of a
        // dtype are one instantiation. A verify row's one-row reference is
        // the decode entry of its dtype, which carries no claim of its own.
        let invariance = match entry.phase {
            AttentionPhase::Decode => continue,
            AttentionPhase::Verify => Invariance {
                rows: one_row,
                ..Invariance::NONE
            },
            AttentionPhase::Prefill => Invariance {
                chunk: Chunk::Any,
                ..Invariance::NONE
            },
        };
        declared.push((entry.symbol(), invariance));
    }
    // The partial merge folds the ranks of a (row, head) in rank order; the
    // localization maps each row's positions alone.
    for entry in group_entries(shape) {
        declared.push((entry.symbol(), rows));
    }
    let profile = Some(profile(shape));
    declared
        .into_iter()
        .map(|(symbol, invariance)| Declared {
            symbol,
            profile,
            op: OpKind::ContextParallel,
            invariance,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The group buffers' neutral sizes are the partials' and merge's
    /// geometry at every geometry and served topology.
    #[test]
    fn dims_are_the_partials_geometry() {
        for geometry in crate::geometry::GEOMETRIES {
            for topology in crate::contract::TOPOLOGIES {
                let shape = Shape::new(geometry, topology.ranks);
                let partials = partials_geometry(shape);
                let merge = merge_geometry(shape);
                let dims = dims(shape);
                assert_eq!(dims.heads, partials.q_heads * partials.kv_heads);
                assert_eq!(dims.heads, merge.heads);
                assert_eq!(dims.width, partials.head_dim);
                assert_eq!(dims.kept, partials.top_blocks);
                assert_eq!(dims.scratch_rows, merge.capacity);
            }
        }
    }

    #[test]
    fn only_tp4_serves_size_two() {
        assert_eq!(sizes(Shape::TP4), [2]);
        assert!(sizes(Shape::of(Ranks::TP2).unwrap()).is_empty());
        assert!(rank_bytes(Shape::of(Ranks::TP2).unwrap(), 2).is_err());
        assert!(rank_bytes(Shape::TP4, 4).is_err());
        // Any other geometry, such as the geometry recipe's data-only
        // additions (synthetic-wide's eight indexer heads have no head-sum
        // entry), serves size 1 only; tests/model-geometry-recipe.sh builds
        // and serves them.
        assert_eq!(GEOMETRIES, [crate::geometry::FLASH.id]);
        assert!(has_entries(Shape::TP4));
    }

    /// The entries take QSA's TP4 rank: six Q heads over one KV head of
    /// 256, the four-head indexer of 128 in blocks of 4 keeping 512 blocks.
    #[test]
    fn entries_follow_the_rank_geometry() {
        let shape = Shape::TP4;
        assert_eq!(
            selection(shape),
            head_sum::Geometry {
                heads: 4,
                head_dim: 128,
                group: 4,
                page_size: 64,
                max_top_k: 2048,
                always_tail: true,
            }
        );
        let g = merge_geometry(shape);
        assert_eq!(
            (g.heads, g.width, g.page_size, g.list_width, g.capacity),
            (6, 256, 64, 1, 8192)
        );
        assert_eq!(
            symbols(shape),
            [
                "chop_owned_writes_qwen38_kv_write_bf16_v1",
                "chop_owned_writes_qwen38_kv_write_fp8_v1",
                "chop_owned_writes_qwen38_index_key_write_v1",
                "chop_pooled_indexer_qwen38_head_sum_select_per_rank2_v1",
                "chop_pooled_indexer_qwen38_head_sum_select_per_rank2_merge_v1",
                "chop_pooled_indexer_qwen38_topk_merge_blocks2_v1",
                "chop_selected_partials_qwen38_bf16_decode_v1",
                "chop_selected_partials_qwen38_bf16_verify_v1",
                "chop_selected_partials_qwen38_bf16_prefill_v1",
                "chop_selected_partials_qwen38_fp8_decode_v1",
                "chop_selected_partials_qwen38_fp8_verify_v1",
                "chop_selected_partials_qwen38_fp8_prefill_v1",
                "chop_context_parallel_qwen38_partial_merge2_v1",
                "chop_context_parallel_qwen38_localize_positions2_v1",
            ]
        );
        let specs = cubin_specs(shape).unwrap();
        let names = specs.iter().map(|s| s.name.clone()).collect::<Vec<_>>();
        assert_eq!(names, cubin_names(shape));
        assert!(names.iter().all(|name| is_cubin(name)));
        let built = specs
            .iter()
            .flat_map(|spec| spec.entries.iter().map(|entry| entry.symbol.clone()))
            .collect::<Vec<_>>();
        assert_eq!(built, symbols(shape));
        select_entry(shape).check_top_k(2048).unwrap();
        select_entry(shape).check_context_parallel(1, 2).unwrap();
        localize_entry(shape).check_rank(1).unwrap();
        assert_eq!(merge_entry(shape).width(), 512);
    }

    /// The catalog lists the TP4 group's entries and cubins, and a group
    /// whose cubins miss an entry is refused.
    #[test]
    fn the_catalog_lists_the_group() {
        let groups = (crate::sm120::CATALOG.context_parallel)();
        let [group] = &groups[..] else {
            panic!("one context-parallel group");
        };
        assert_eq!((group.profile, group.ranks), (profile(Shape::TP4), RANKS));
        assert_eq!(group.symbols(), symbols(Shape::TP4));
        let names = group
            .cubins
            .iter()
            .map(|c| c.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(names, cubin_names(Shape::TP4));
        group.check().unwrap();
        let mut short = group.clone();
        short.cubins[0].entries.pop();
        assert!(short.check().is_err());
        let cubins = kernels_sm120::context_parallel::cubin_specs(crate::sm120::catalogs(), None);
        assert_eq!(cubins.len(), names.len());
    }

    /// The verify partials of each dtype are the decode partials'
    /// instantiation under another symbol: the same body, parameters and
    /// threads, so the decode entry is the verify entry's one-row launch.
    #[test]
    fn verify_partials_share_the_decode_instantiation() {
        let entries = partial_entries(Shape::TP4);
        for verify in entries.iter().filter(|e| e.phase == AttentionPhase::Verify) {
            let decode = entries
                .iter()
                .find(|e| e.phase == AttentionPhase::Decode && e.kv == verify.kv)
                .unwrap();
            let (a, b) = (verify.entry_spec().unwrap(), decode.entry_spec().unwrap());
            assert_ne!(a.symbol, b.symbol);
            assert_eq!(
                (&a.params, &a.body, a.max_threads, a.min_blocks),
                (&b.params, &b.body, b.max_threads, b.min_blocks)
            );
        }
    }

    /// The declarations cover every symbol but the selection entry (whose
    /// output is split workspace) and the decode partials, on the TP4
    /// profile, and join the catalog's.
    #[test]
    fn declarations_cover_the_family_but_the_selection_and_decode_partials() {
        let shape = Shape::TP4;
        let declared = declared(shape);
        let mut undeclared = partial_entries(shape)
            .into_iter()
            .filter(|e| e.phase == AttentionPhase::Decode)
            .map(|e| e.symbol())
            .collect::<Vec<_>>();
        assert_eq!(undeclared.len(), 2);
        undeclared.push(select_entry(shape).symbol());
        let expected = symbols(shape)
            .into_iter()
            .filter(|symbol| !undeclared.contains(symbol))
            .collect::<Vec<_>>();
        let symbols = declared
            .iter()
            .map(|d| d.symbol.clone())
            .collect::<Vec<_>>();
        assert_eq!(symbols, expected);
        assert!(
            declared
                .iter()
                .all(|d| { d.op == OpKind::ContextParallel && d.profile == Some(profile(shape)) })
        );
        let catalog = (crate::sm120::CATALOG.declared)();
        for d in &declared {
            let found = catalog.iter().find(|c| c.symbol == d.symbol).unwrap();
            assert_eq!((found.op, found.invariance), (d.op, d.invariance));
        }
        assert!(!catalog.iter().any(|c| undeclared.contains(&c.symbol)));
    }

    /// Every declaration provides what the bundle's context-parallel ops
    /// ask at group size 2: one-row bits at the exact verify rows, and at
    /// prefill chunks of the prefill granule with anchors (the key write
    /// keeps its ring at a spare entry for anchors past the granule's rows;
    /// the others keep no per-slot state).
    /// The verify partials are asked the verify need only, the prefill
    /// partials the prefill need only. The selection op is asked through
    /// its split merge's claim, the composition's.
    #[test]
    fn declarations_provide_the_exact_verify_and_prefill_needs() {
        let verify = Invariance {
            rows: Rows::OneRow {
                max: crate::contract::EXACT_VERIFY.rows,
            },
            ..Invariance::NONE
        };
        let prefill = |anchors| Invariance {
            chunk: Chunk::Granule(crate::contract::PREFILL_GRANULE),
            anchors,
            ..Invariance::NONE
        };
        let shape = Shape::TP4;
        let key_write = owned_entries(shape)
            .into_iter()
            .find(|e| e.kind == owned_writes::Kind::KeyWrite)
            .unwrap()
            .symbol();
        let phase = |symbol: &str| {
            partial_entries(shape)
                .into_iter()
                .find(|e| e.symbol() == symbol)
                .map(|e| e.phase)
        };
        for d in declared(shape) {
            let anchors = if d.symbol == key_write {
                Anchors::SpareAfter {
                    rows: crate::contract::PREFILL_GRANULE,
                }
            } else {
                Anchors::NotApplicable
            };
            let (at_verify, at_prefill) = match phase(&d.symbol) {
                Some(AttentionPhase::Verify) => (true, false),
                Some(AttentionPhase::Prefill) => (false, true),
                Some(AttentionPhase::Decode) => panic!("{}: decode declared", d.symbol),
                None => (true, true),
            };
            assert!(!at_verify || d.invariance.provides(verify), "{}", d.symbol);
            assert!(
                !at_prefill || d.invariance.provides(prefill(anchors)),
                "{}",
                d.symbol
            );
            assert!(
                d.invariance
                    .anchors
                    .supported_after(crate::contract::PREFILL_GRANULE),
                "{}",
                d.symbol
            );
        }
    }

    /// The owned key write's spare ring is complete only for anchors a ring
    /// or more after the launch's start: the backend, with the model's
    /// claims registered, refuses a need of anchors at any span (four live
    /// rows after 4,096 cached tokens anchored at 4,100, with or without a
    /// chunk or rows claim) and grants the bundle's prefill need.
    #[test]
    fn the_owned_key_write_refuses_anchors_inside_the_ring() {
        use kernel_core::backend::{Backend, Need, Refused};
        use kernel_core::context_parallel::OwnedKeyWriteOp;
        let shape = Shape::TP4;
        let key_write = owned_entries(shape)
            .into_iter()
            .find(|e| e.kind == owned_writes::Kind::KeyWrite)
            .unwrap();
        let backend = kernels_sm120::backend::Sm120::new()
            .with_owned_writes(owned_entries(shape))
            .with_context_parallel_claims(
                declared(shape)
                    .into_iter()
                    .map(|d| (d.symbol, d.invariance)),
            );
        let op = OwnedKeyWriteOp {
            query: key_write.query(),
            index_heads: key_write.indexer.index_heads,
            max_context: key_write.indexer.max_context,
        };
        let need = |chunk, rows, anchors| {
            Need::rows(4).with_exact(Invariance {
                chunk,
                rows,
                anchors,
                ..Invariance::NONE
            })
        };
        let four = Rows::OneRow { max: 4 };
        for (chunk, rows) in [
            (Chunk::None, Rows::None),
            (Chunk::None, four),
            (Chunk::Granule(4), Rows::None),
            (Chunk::Granule(crate::contract::PREFILL_GRANULE), four),
        ] {
            assert!(matches!(
                backend.owned_key_write(&op, 1, need(chunk, rows, Anchors::Spare)),
                Err(Refused::Inexact { .. })
            ));
        }
        let past = |rows| Anchors::SpareAfter { rows };
        assert!(matches!(
            backend.owned_key_write(&op, 1, need(Chunk::None, Rows::None, past(4))),
            Err(Refused::Inexact { .. })
        ));
        let served = need(
            Chunk::Granule(crate::contract::PREFILL_GRANULE),
            four,
            past(crate::contract::PREFILL_GRANULE),
        );
        let prepared = backend.owned_key_write(&op, 1, served).unwrap();
        assert_eq!(
            prepared.info().guarantees.anchors,
            past(key_write.indexer.ring)
        );
    }

    /// One group rank's context-parallel launches, in the order a DCP2
    /// layer runs them, through the `phase` partials of `kv`.
    fn lane(shape: Shape, phase: AttentionPhase, kv: kernel_core::contract::DType) -> Vec<String> {
        let owned = owned_entries(shape);
        let owned =
            |kind: owned_writes::Kind| owned.iter().find(|e| e.kind == kind).unwrap().symbol();
        let partials = partial_entries(shape)
            .into_iter()
            .find(|e| e.phase == phase && e.kv == kv)
            .unwrap()
            .symbol();
        let select = select_entry(shape);
        let gather = ALLGATHER_SYMBOL.to_owned();
        vec![
            localize_entry(shape).symbol(),
            owned(owned_writes::Kind::KvWrite { kv }),
            owned(owned_writes::Kind::KeyWrite),
            select.symbol(),
            select.merge_symbol().unwrap(),
            gather.clone(),
            merge_entry(shape).symbol(),
            gather.clone(),
            partials.clone(),
            partials,
            gather.clone(),
            gather.clone(),
            partial_merge_entry(shape).symbol(),
            gather,
        ]
    }

    /// A DCP2 lane passes the program's journal check at an exact verify
    /// step and at prefill: every launch's declaration, and the head-sum
    /// selection through its split merge's (the composition's). A
    /// selection its merge does not follow is refused.
    #[test]
    fn a_group_lane_passes_the_journal_check() {
        use kernel_core::contract::DType;
        use kernels_sm120::invariance::{Launched, Program};
        let program = Program {
            prefill_granule: crate::contract::PREFILL_GRANULE,
            exact_verify: crate::contract::EXACT_VERIFY,
            anchor_items: crate::contract::ANCHOR_ITEMS,
        };
        let rows = crate::contract::EXACT_VERIFY.rows;
        let verify = Launched::Verify {
            rows,
            items: 1,
            launch_rows: bucket(rows),
        };
        let shape = Shape::TP4;
        let catalogs = crate::sm120::catalogs();
        for (launched, phase) in [
            (verify, AttentionPhase::Verify),
            (Launched::Prefill, AttentionPhase::Prefill),
        ] {
            for kv in [DType::Bf16, DType::Fp8E4M3] {
                let lane = lane(shape, phase, kv);
                let check = |lane: &[String]| {
                    program.check_journal(
                        catalogs,
                        lane.iter().map(|s| (launched, s.as_str())),
                        &[],
                    )
                };
                assert_eq!(check(&lane), Vec::<String>::new(), "{phase:?} {kv:?}");
                let mut unfollowed = lane.clone();
                unfollowed.remove(4);
                let violations = check(&unfollowed);
                assert_eq!(violations.len(), 1, "{phase:?} {kv:?}: {violations:?}");
                assert!(violations[0].contains("not followed"));
            }
        }
    }

    /// The group all-gather is the served comm cubin's entry, so the family
    /// holds no copy of it.
    #[test]
    fn the_group_gather_is_the_served_comm_entry() {
        let comm = kernels_sm120::comm::cubin_spec();
        assert!(
            comm.entries
                .iter()
                .any(|entry| entry.symbol == ALLGATHER_SYMBOL)
        );
        assert!(!symbols(Shape::TP4).iter().any(|s| s == ALLGATHER_SYMBOL));
    }

    /// One rank's buffers at size 2: two send and two received blocks of
    /// 2048 rows of six heads, the gathered queries, the pairs of every
    /// bucket and the selection's widest split workspace.
    #[test]
    fn rank_bytes_follow_the_bucket() {
        let dims = dims(Shape::TP4);
        assert_eq!(
            (dims.heads, dims.width, dims.kept, dims.scratch_rows),
            (6, 256, 512, 8192)
        );
        assert_eq!(buckets().count(), 12);
        assert_eq!(bucket_index(24).unwrap(), 5);
        assert!(bucket_index(MAX_BUCKET + 1).is_err());
        let bytes = rank_bytes(Shape::TP4, 2).unwrap();
        let block = 2048 * 6 * 258 * 4;
        assert_eq!(bytes.send, 2 * block);
        assert_eq!(bytes.receive, 2 * block);
        assert_eq!(bytes.queries, 2 * 2048 * 6 * 256 * 2);
        assert_eq!(bytes.gathered_pairs[11], 2 * 2048 * 1024 * 4);
        assert_eq!(bytes.pairs, 2048 * 1024 * 4);
        assert_eq!(bytes.local, 2048 * 512 * 8);
        assert_eq!(bytes.barrier_output, 2 * 4);
        assert_eq!(bytes.total(), RANK_BYTES_AT_2);
    }

    /// [`RankBytes::total`] at size 2: about 0.11 GiB per rank.
    const RANK_BYTES_AT_2: u64 = 113_731_012;
}
