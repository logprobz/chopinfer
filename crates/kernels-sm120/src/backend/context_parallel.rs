//! Context parallelism through the entries a model instantiates: the
//! partial merge and key localization (`context_parallel`), the paged
//! partials (`paged_partials`) and the owner-routed writes
//! (`owned_writes`), and the group transfers through the shared all-gather
//! (`context_parallel::group`) or the model's phased group gather
//! (`context_parallel::phased`).
//!
//! An op is served by the entry whose query equals the op's apart from the
//! capacity and that holds `need.capacity` rows, the least capacity first
//! (`select::choose`); registration order never decides. An op guarantees
//! what the model declares for its entry (`Claims`, by symbol), and nothing
//! for an entry the model declares nothing for. A group transfer guarantees
//! what the comm declarations claim for its transport.
//! Each launch plans the entry for the step's rows, a graph bucket whose
//! rows past the device live count return, as the models' builders plan
//! it.
//!
//! A group transfer's transport is SM120's choice, and every transport
//! moves the same bits to the same places. Without a phased gather in the
//! selection, gathers, owner gathers and barriers run the shared all-gather
//! and an exchange is refused. With one, gathers and exchanges run it (one
//! cluster that sends to one member at a time) and owner gathers and
//! barriers stay on the shared all-gather.

use kernel_build::EntryParam;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::context_parallel::{
    ContextParallelQuery, GroupTransfer, GroupTransferBuffers, GroupTransferOp,
    LocalizeKeysBuffers, OwnedKeyWriteOp, OwnedKvWriteOp, PagedPartialsBuffers, PagedPartialsOp,
    PartialMergeBuffers,
};
use kernel_core::contract::{
    Argument, ArgumentList, DType, GatedAttentionQuery, LaunchPlan, Refusal, SparseAttentionQuery,
    Validate,
};
use kernel_core::gated_sparse::{GatedKvWriteBuffers, KvCache, SparseKeyWriteBuffers};
use kernel_core::properties::Invariance;
use std::collections::BTreeMap;

use super::gated_sparse::{check_block_keys, check_cache, check_rings, check_table};
use super::select::{self, Candidate};
use super::{MAX_CAPACITY, check_plan};
use crate::context_parallel::group::{self, GatherState, GroupRank};
use crate::context_parallel::{self as cp, Kind, MergeLayout, phased};
use crate::{owned_writes, paged_partials};

/// A model's context-parallel entries.
#[derive(Clone, Debug, Default)]
pub(super) struct Selection {
    /// Partial merges and key localizations (`context_parallel::entries`).
    pub(super) entries: Vec<cp::Entry>,
    /// Paged partials (`paged_partials::entries`).
    pub(super) paged: Vec<paged_partials::Entry>,
    /// Owner-routed writes (`owned_writes::entries`).
    pub(super) owned: Vec<owned_writes::Entry>,
    /// Whether the model instantiates the phased group gather.
    pub(super) phased: bool,
    /// The guarantees the model declares for its entries.
    pub(super) claims: Claims,
}

/// The guarantees a model declares for its context-parallel entries, by
/// symbol (`Sm120::with_context_parallel_claims`): partial merges, key
/// localizations and owner-routed writes here, selected partials and
/// head-sum selections, their split merges and block-id merges in their
/// families. The entries carry no claims of their own, so their records,
/// launch keys and implementation names stay as they are.
#[derive(Clone, Debug, Default)]
pub(super) struct Claims(BTreeMap<String, Invariance>);

impl Claims {
    /// Adds `symbol`'s claims.
    ///
    /// # Panics
    ///
    /// On a symbol claimed twice with different guarantees.
    pub(super) fn insert(&mut self, symbol: String, invariance: Invariance) {
        if let Some(held) = self.0.insert(symbol.clone(), invariance) {
            assert_eq!(held, invariance, "{symbol} is claimed twice differently");
        }
    }

    /// What an op launching every symbol of `symbols` guarantees: the meet
    /// of their claims, nothing if one has none.
    pub(super) fn of(&self, symbols: &[&str]) -> Invariance {
        self.of_or(symbols, Invariance::NONE)
    }

    /// [`Claims::of`], with `undeclared` for a symbol without claims.
    pub(super) fn of_or(&self, symbols: &[&str], undeclared: Invariance) -> Invariance {
        symbols
            .iter()
            .map(|symbol| self.0.get(*symbol).copied().unwrap_or(undeclared))
            .reduce(Invariance::meet)
            .unwrap_or(undeclared)
    }
}

/// What an undeclared entry that writes per-slot state (a key ring)
/// guarantees: nothing, and no spare entry at anchors. `Invariance::NONE`
/// says it writes no per-slot state (`Anchors::NotApplicable`), which would
/// grant any anchor need.
pub(super) const STATEFUL_NONE: Invariance = Invariance {
    anchors: kernel_core::properties::Anchors::None,
    ..Invariance::NONE
};

/// The partial merge of `op` through the selection's entry for it.
pub(super) fn partial_merge(
    selection: &Selection,
    op: &ContextParallelQuery,
    need: Need,
) -> Result<Prepared<PartialMergeBuffers<View>>, Refused> {
    op.validate()?;
    if !matches!(op, ContextParallelQuery::PartialMerge { .. }) {
        return Err(Refused::Rule(
            "a partial merge merges by a partial merge query".into(),
        ));
    }
    let (entry, info) = choose_cp(selection, "partial merge", op, need)?;
    Ok(Box::new(PartialMerge {
        entry,
        capacity: need.capacity,
        info,
    }))
}

/// The key localization of `op` to group rank `group_rank` through the
/// selection's entry for it.
pub(super) fn localize_keys(
    selection: &Selection,
    op: &ContextParallelQuery,
    group_rank: u32,
    need: Need,
) -> Result<Prepared<LocalizeKeysBuffers<View>>, Refused> {
    op.validate()?;
    if !matches!(op, ContextParallelQuery::LocalizeKeys { .. }) {
        return Err(Refused::Rule(
            "a key localization localizes by a localization query".into(),
        ));
    }
    let (entry, info) = choose_cp(selection, "key localization", op, need)?;
    entry.check_rank(group_rank)?;
    Ok(Box::new(LocalizeKeys {
        entry,
        capacity: need.capacity,
        group_rank,
        info,
    }))
}

/// The paged partials of `op` at group rank `group_rank` of `group_size`
/// through the selection's entry for it.
pub(super) fn paged_partials(
    selection: &Selection,
    op: &PagedPartialsOp,
    group_rank: u32,
    group_size: u32,
    need: Need,
) -> Result<Prepared<PagedPartialsBuffers<View>>, Refused> {
    op.validate()?;
    let candidates = selection
        .paged
        .iter()
        .filter(|entry| uncapped(entry.query()) == uncapped(op.query))
        .map(|entry| {
            candidate(
                entry,
                entry.symbol(),
                paged_launch(entry),
                entry.capacity,
                &selection.claims,
                Invariance::NONE,
            )
        })
        .collect::<Vec<_>>();
    let (entry, guarantees) = choose("paged partials", candidates, need)?;
    entry.contract()?;
    entry.check_rank(group_rank, group_size)?;
    let symbol = entry.symbol();
    let info = info(
        "paged_partials",
        &entry,
        &symbol,
        entry.plan(need.capacity)?,
        guarantees,
        need,
    )?;
    Ok(Box::new(PagedPartials {
        max_pages: op.max_context.div_ceil(entry.geometry.page),
        entry,
        capacity: need.capacity,
        scale: op.softmax_scale,
        group: [group_rank, group_size],
        info,
    }))
}

/// The owner-routed KV write of `op` through the selection's entry for it.
pub(super) fn owned_kv_write(
    selection: &Selection,
    op: &OwnedKvWriteOp,
    need: Need,
) -> Result<Prepared<GatedKvWriteBuffers<View>>, Refused> {
    op.validate()?;
    let (entry, guarantees) = choose_owned(
        selection,
        "owner-routed KV write",
        op.query,
        need,
        |entry| matches!(entry.kind, owned_writes::Kind::KvWrite { .. }),
    )?;
    let ContextParallelQuery::OwnedQkNormRopeKvWrite {
        write: GatedAttentionQuery::QkNormRopeKvWrite { page_size, .. },
        ..
    } = op.query
    else {
        unreachable!("a validated owner-routed KV write holds a KV write query");
    };
    let symbol = entry.symbol();
    let info = info(
        "owned_writes",
        &entry,
        &symbol,
        entry.plan(need.capacity)?,
        guarantees,
        need,
    )?;
    Ok(Box::new(OwnedKvWrite {
        entry,
        capacity: need.capacity,
        max_pages: op.max_context.div_ceil(page_size),
        info,
    }))
}

/// The owner-routed key write of `op` at group rank `group_rank` through
/// the selection's entry for it.
pub(super) fn owned_key_write(
    selection: &Selection,
    op: &OwnedKeyWriteOp,
    group_rank: u32,
    need: Need,
) -> Result<Prepared<SparseKeyWriteBuffers<View>>, Refused> {
    op.validate()?;
    let (entry, guarantees) = choose_owned(
        selection,
        "owner-routed key write",
        op.query,
        need,
        |entry| {
            entry.kind == owned_writes::Kind::KeyWrite
                && entry.indexer.index_heads == op.index_heads
                && entry.indexer.max_context == op.max_context
        },
    )?;
    entry.check_rank(group_rank, entry.ranks)?;
    let symbol = entry.symbol();
    let info = info(
        "owned_writes",
        &entry,
        &symbol,
        entry.plan(need.capacity)?,
        guarantees,
        need,
    )?;
    Ok(Box::new(OwnedKeyWrite {
        max_pages: entry.indexer.max_context.div_ceil(entry.indexer.page_size),
        entry,
        capacity: need.capacity,
        group_rank,
        info,
    }))
}

/// The group transfer `op` that group rank `group_rank` launches, through
/// the transport the selection chooses.
pub(super) fn group_transfer(
    selection: &Selection,
    op: &GroupTransferOp,
    group_rank: u32,
    need: Need,
) -> Result<Prepared<GroupTransferBuffers<View>>, Refused> {
    op.validate()?;
    check_capacity(need)?;
    let rank = GroupRank::of(group_rank, op.size, op.size)?;
    let transport = match (op.transfer, selection.phased) {
        (GroupTransfer::Gather | GroupTransfer::Exchange, true) => Transport::Phased,
        (GroupTransfer::Exchange, false) => {
            return Err(Refused::Rule(
                "an SM120 group exchange runs the phased group gather, which the selection lacks"
                    .into(),
            ));
        }
        _ => Transport::Shared,
    };
    let (symbol, plan, state) = match transport {
        Transport::Shared => {
            let buffers = group::gather_buffers(op.size, op.words)?;
            (
                group::ALLGATHER_SYMBOL,
                group::gather_plan(op.words)?,
                vec![buffers.flag_bytes, buffers.epoch_bytes],
            )
        }
        Transport::Phased => (
            phased::SYMBOL,
            phased::plan(),
            vec![phased::flag_bytes(op.size), phased::EPOCH_BYTES],
        ),
    };
    check_plan(&plan)?;
    // Every transport moves bits; a transfer claims what the comm
    // declarations claim for its entry, nothing for an undeclared one.
    let guarantees = crate::comm::declared_entries()
        .into_iter()
        .find(|declared| declared.symbol == symbol)
        .map_or(Invariance::NONE, |declared| declared.invariance);
    let mut info = OpInfo::new(
        format!("sm120.context_parallel.{}.v1[{symbol}]", transport.name()),
        guarantees,
        vec![(symbol.to_owned(), 0)],
    );
    info.state = state;
    need.check(&info)?;
    Ok(Box::new(Transfer {
        op: *op,
        rank,
        transport,
        capacity: need.capacity,
        plan,
        info,
    }))
}

fn check_capacity(need: Need) -> Result<(), Refused> {
    if need.capacity == 0 || need.capacity > MAX_CAPACITY {
        return Err(Refused::Rule(format!(
            "an SM120 context-parallel op serves 1 to {MAX_CAPACITY} rows"
        )));
    }
    Ok(())
}

/// `query` with its capacity cleared: the entry's static shape.
fn uncapped(query: ContextParallelQuery) -> ContextParallelQuery {
    let mut query = query;
    match &mut query {
        ContextParallelQuery::PartialMerge { capacity, .. }
        | ContextParallelQuery::TopKMerge { capacity, .. }
        | ContextParallelQuery::LocalizeKeys { capacity, .. }
        | ContextParallelQuery::PagedAttentionPartials { capacity, .. }
        | ContextParallelQuery::SelectedAttentionPartials { capacity, .. }
        | ContextParallelQuery::TopKMergeBlocks { capacity, .. } => *capacity = 0,
        ContextParallelQuery::OwnedQkNormRopeKvWrite {
            write: GatedAttentionQuery::QkNormRopeKvWrite { capacity, .. },
            ..
        }
        | ContextParallelQuery::OwnedIndexerKeyWrite {
            write: SparseAttentionQuery::IndexerKeyWrite { capacity, .. },
            ..
        } => *capacity = 0,
        ContextParallelQuery::OwnedQkNormRopeKvWrite { .. }
        | ContextParallelQuery::OwnedIndexerKeyWrite { .. } => {}
    }
    query
}

/// An entry that matches an op, with the claims the model declares for it
/// (`undeclared` without).
fn candidate<E: std::fmt::Debug>(
    entry: E,
    symbol: String,
    launches: String,
    capacity: u32,
    claims: &Claims,
    undeclared: Invariance,
) -> Candidate<E> {
    let guarantees = claims.of_or(&[&symbol], undeclared);
    Candidate::new(entry, symbol, launches, guarantees).capacity(capacity)
}

/// The entry of `candidates` that serves the op at `need`, and its
/// guarantees (`select::choose`).
fn choose<E: Clone>(
    op: &str,
    candidates: Vec<Candidate<&E>>,
    need: Need,
) -> Result<(E, Invariance), Refused> {
    check_capacity(need)?;
    let chosen = select::choose(op, candidates, need)?;
    Ok((chosen.entry.clone(), chosen.guarantees))
}

/// What a merge or localization entry launches: its symbol and the
/// instantiation its kind compiles (its ranks and the geometry it reads),
/// without the capacity the selection settles.
fn cp_launch(entry: &cp::Entry) -> String {
    let cp::Entry {
        profile: _,  // filters catalogs; no launch reads it
        name: _,     // in the symbol
        geometry: _, // read through the query; selects the op
        kind: _,     // read through the query; selects the op
    } = entry;
    format!("{} {:?}", entry.symbol(), uncapped(entry.query()))
}

/// What a paged partials entry launches: its symbol and its geometry, KV
/// dtype and phase, without the capacity the selection settles.
fn paged_launch(entry: &paged_partials::Entry) -> String {
    let paged_partials::Entry {
        profile: _,  // filters catalogs; no launch reads it
        name: _,     // in the symbol
        geometry,    // the query's; selects the op
        kv,          // the query's; selects the op
        phase,       // the query's; selects the op
        capacity: _, // bounds the rows the selection settles
    } = entry;
    format!("{} {geometry:?} {kv:?} {phase:?}", entry.symbol())
}

/// What an owner-routed write launches: its symbol, the instantiation its
/// query compiles without the capacity the selection settles and, for a
/// key write, the indexer traits its body names beyond the query (its
/// heads and top-k).
fn owned_launch(entry: &owned_writes::Entry) -> String {
    let owned_writes::Entry {
        profile: _, // filters catalogs; no launch reads it
        name: _,    // in the symbol
        kind,       // selects the op
        gated: _,   // a KV write's query; selects the op
        indexer,    // a key write's query, heads and top-k
        ranks: _,   // the query's; selects the op
    } = entry;
    let query = uncapped(entry.query());
    match kind {
        owned_writes::Kind::KvWrite { .. } => format!("{} {query:?}", entry.symbol()),
        owned_writes::Kind::KeyWrite => format!(
            "{} {query:?} {} {}",
            entry.symbol(),
            indexer.index_heads,
            indexer.top_blocks
        ),
    }
}

/// The merge or localization entry serving `op`, with its info. A group
/// past SM120's limit is refused before any entry is looked up.
fn choose_cp(
    selection: &Selection,
    family: &str,
    op: &ContextParallelQuery,
    need: Need,
) -> Result<(cp::Entry, OpInfo), Refused> {
    cp::check_query(op)?;
    let candidates = selection
        .entries
        .iter()
        .filter(|entry| uncapped(entry.query()) == uncapped(*op))
        .map(|entry| {
            candidate(
                entry,
                entry.symbol(),
                cp_launch(entry),
                entry.geometry.capacity,
                &selection.claims,
                Invariance::NONE,
            )
        })
        .collect::<Vec<_>>();
    let (entry, guarantees) = choose(family, candidates, need)?;
    entry.contract()?;
    let symbol = entry.symbol();
    let info = info(
        "context_parallel",
        &entry,
        &symbol,
        entry.plan(need.capacity)?,
        guarantees,
        need,
    )?;
    Ok((entry, info))
}

/// The owner-routed write entry of `query` that `serves` the op, and its
/// guarantees, refusing a group past SM120's limit first.
fn choose_owned(
    selection: &Selection,
    family: &str,
    query: ContextParallelQuery,
    need: Need,
    serves: impl Fn(&owned_writes::Entry) -> bool,
) -> Result<(owned_writes::Entry, Invariance), Refused> {
    cp::check_query(&query)?;
    let candidates = selection
        .owned
        .iter()
        .filter(|entry| serves(entry) && uncapped(entry.query()) == uncapped(query))
        .map(|entry| {
            candidate(
                entry,
                entry.symbol(),
                owned_launch(entry),
                entry.capacity(),
                &selection.claims,
                // The key write keeps key rings; the K/V write writes paged
                // KV only.
                match entry.kind {
                    owned_writes::Kind::KeyWrite => STATEFUL_NONE,
                    owned_writes::Kind::KvWrite { .. } => Invariance::NONE,
                },
            )
        })
        .collect::<Vec<_>>();
    let chosen = choose(family, candidates, need)?;
    chosen.0.contract()?;
    Ok(chosen)
}

/// The info of one-entry op `entry` of `module`, whose widest plan is
/// `plan` and which keeps `guarantees`: its implementation names the
/// entry's whole record.
fn info(
    module: &str,
    entry: &impl std::fmt::Debug,
    symbol: &str,
    plan: LaunchPlan,
    guarantees: Invariance,
    need: Need,
) -> Result<OpInfo, Refused> {
    check_plan(&plan)?;
    let digest = kernel_build::digest(format!("{entry:?}").as_bytes());
    let info = OpInfo::new(
        format!("sm120.{module}.v1[{symbol}]#{}", &digest[..16]),
        guarantees,
        vec![(symbol.to_owned(), plan.dynamic_shared_bytes)],
    );
    need.check(&info)?;
    Ok(info)
}

fn buffer(view: View) -> Argument {
    Argument::DeviceBuffer(view.handle)
}

/// The product of `factors` bytes, refused rather than wrapped.
fn bytes(factors: &[u64]) -> Result<u64, Refusal> {
    factors
        .iter()
        .try_fold(1u64, |product, &factor| product.checked_mul(factor))
        .ok_or(Refusal("context-parallel bytes overflow"))
}

/// Packs `values` for an entry of parameters `params`, refusing values of
/// other kinds.
fn pack(params: &[EntryParam], values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
    if values.len() != params.len()
        || !params.iter().zip(&values).all(|(param, value)| {
            matches!(
                (param, value),
                (EntryParam::DevicePointer { .. }, Argument::DeviceBuffer(_))
                    | (EntryParam::U32 { .. }, Argument::U32(_))
                    | (EntryParam::U64 { .. }, Argument::U64(_))
            )
        })
    {
        return Err(Refusal("context-parallel argument ABI mismatch"));
    }
    let arguments = ArgumentList(values);
    arguments.validate()?;
    Ok(arguments)
}

fn launch(symbol: String, rows: u32, plan: LaunchPlan, arguments: ArgumentList) -> Launch {
    Launch {
        symbol,
        rows,
        plan,
        arguments,
        collective: false,
        persistent: false,
    }
}

/// The step's rows, refused outside the prepared capacity.
fn rows(counts: Counts, capacity: u32) -> Result<u64, Refusal> {
    if counts.rows == 0 || counts.rows > capacity {
        return Err(Refusal("row count outside the prepared capacity"));
    }
    Ok(u64::from(counts.rows))
}

/// The items a step's per-item arrays hold: at least one.
fn items(counts: Counts) -> u64 {
    u64::from(counts.items.max(1))
}

fn scale_bits(scale: f32) -> Argument {
    Argument::U32(scale.to_bits())
}

/// The cache of `heads` KV heads of `dim` in `kv`, `page` tokens a page.
fn check_kv(
    cache: &KvCache<View>,
    page: u32,
    heads: u32,
    dim: u32,
    kv: DType,
) -> Result<(), Refusal> {
    let fp8 = kv == DType::Fp8E4M3;
    let element = if fp8 { 1 } else { 2 };
    check_cache(
        cache,
        bytes(&[u64::from(page), u64::from(heads), u64::from(dim), element])?,
        fp8,
    )
}

/// `partial_merge`: a CTA per row and head.
struct PartialMerge {
    entry: cp::Entry,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<PartialMergeBuffers<View>> for PartialMerge {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &PartialMergeBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let Kind::PartialMerge { ranks } = self.entry.kind else {
            unreachable!("a prepared partial merge holds a merge entry");
        };
        let layout = MergeLayout {
            rank_rows: buffers.rank_rows,
            input_heads: buffers.input_heads,
            head_offset: buffers.head_offset,
        };
        self.entry.check_merge(counts.rows, layout)?;
        let g = self.entry.geometry;
        // Each segment holds `m`, `l` and `o`: `width + 2` floats a pair
        // (`Entry::segment_floats`), every factor checked, since the layout
        // comes from the buffers.
        buffers.partials.covers(bytes(&[
            u64::from(ranks),
            u64::from(layout.rank_rows),
            u64::from(layout.input_heads),
            u64::from(g.width) + 2,
            4,
        ])?)?;
        buffers.live_rows.covers(4)?;
        buffers
            .output
            .covers(bytes(&[rows, u64::from(g.heads), u64::from(g.width), 2])?)?;
        Ok(vec![launch(
            self.entry.symbol(),
            counts.rows,
            self.entry.plan(counts.rows)?,
            pack(
                &self.entry.params(),
                vec![
                    buffer(buffers.partials),
                    buffer(buffers.live_rows),
                    buffer(buffers.output),
                    Argument::U32(layout.rank_rows),
                    Argument::U32(layout.input_heads),
                    Argument::U32(layout.head_offset),
                ],
            )?,
        )])
    }
}

/// `localize_keys`: a CTA per row and tile of list entries.
struct LocalizeKeys {
    entry: cp::Entry,
    capacity: u32,
    group_rank: u32,
    info: OpInfo,
}

impl PreparedOp<LocalizeKeysBuffers<View>> for LocalizeKeys {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &LocalizeKeysBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let list = bytes(&[rows, u64::from(self.entry.geometry.list_width), 4])?;
        buffers.lists.covers(list)?;
        buffers.output.covers(list)?;
        buffers.live_rows.covers(4)?;
        Ok(vec![launch(
            self.entry.symbol(),
            counts.rows,
            self.entry.plan(counts.rows)?,
            pack(
                &self.entry.params(),
                vec![
                    buffer(buffers.lists),
                    buffer(buffers.live_rows),
                    buffer(buffers.output),
                    Argument::U32(self.group_rank),
                ],
            )?,
        )])
    }
}

/// `paged_partials`: a CTA per row and KV head.
struct PagedPartials {
    entry: paged_partials::Entry,
    capacity: u32,
    max_pages: u32,
    scale: f32,
    group: [u32; 2],
    info: OpInfo,
}

impl PreparedOp<PagedPartialsBuffers<View>> for PagedPartials {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &PagedPartialsBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let g = self.entry.geometry;
        let dim = u64::from(g.head_dim);
        let heads = u64::from(self.entry.heads());
        buffers.q.covers(bytes(&[rows, heads, dim, 2])?)?;
        let cache = &buffers.cache;
        check_kv(cache, g.page, g.kv_heads, g.head_dim, self.entry.kv)?;
        buffers
            .block_tables
            .covers(bytes(&[items(counts), u64::from(self.max_pages), 4])?)?;
        buffers.token_batch.covers(rows * 4)?;
        buffers.positions.covers(rows * 4)?;
        buffers.live_rows.covers(4)?;
        let pairs = bytes(&[rows, heads, 4])?;
        buffers.max.covers(pairs)?;
        buffers.den.covers(pairs)?;
        buffers.num.covers(bytes(&[pairs, dim])?)?;
        let [group_rank, group_size] = self.group;
        Ok(vec![launch(
            self.entry.symbol(),
            counts.rows,
            self.entry.plan(counts.rows)?,
            pack(
                &self.entry.params(),
                vec![
                    buffer(buffers.q),
                    buffer(cache.k),
                    buffer(cache.v),
                    buffer(buffers.block_tables),
                    buffer(buffers.token_batch),
                    buffer(buffers.positions),
                    buffer(buffers.live_rows),
                    buffer(buffers.max),
                    buffer(buffers.den),
                    buffer(buffers.num),
                    Argument::U32(self.max_pages),
                    Argument::U64(cache.k_slot_stride),
                    Argument::U64(cache.v_slot_stride),
                    scale_bits(self.scale),
                    scale_bits(cache.k_scale),
                    scale_bits(cache.v_scale),
                    Argument::U32(group_rank),
                    Argument::U32(group_size),
                ],
            )?,
        )])
    }
}

/// `owned_qk_norm_rope_kv`: the gated KV write's CTA per row, storing K
/// and V only at a row's local position.
struct OwnedKvWrite {
    entry: owned_writes::Entry,
    capacity: u32,
    max_pages: u32,
    info: OpInfo,
}

impl PreparedOp<GatedKvWriteBuffers<View>> for OwnedKvWrite {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &GatedKvWriteBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let owned_writes::Kind::KvWrite { kv } = self.entry.kind else {
            unreachable!("a prepared owner-routed KV write holds a KV write entry");
        };
        let g = self.entry.gated;
        let (q_heads, kv_heads, dim) = (
            u64::from(g.q_heads),
            u64::from(g.kv_heads),
            u64::from(g.head_dim),
        );
        buffers
            .qkv
            .covers(bytes(&[rows, 2 * (q_heads + kv_heads), dim, 2])?)?;
        buffers.q.covers(bytes(&[rows, q_heads, dim, 2])?)?;
        buffers.token_batch.covers(rows * 4)?;
        buffers.positions.covers(rows * 4)?;
        buffers.live_rows.covers(4)?;
        buffers
            .block_tables
            .covers(bytes(&[items(counts), u64::from(self.max_pages), 4])?)?;
        let cache = &buffers.cache;
        check_kv(cache, g.page, g.kv_heads, g.head_dim, kv)?;
        let rope = &buffers.rope;
        let (Some(cos_sin), Some(q_norm), Some(k_norm), Some(rope_positions)) =
            (rope.cos_sin, rope.q_norm, rope.k_norm, rope.rope_positions)
        else {
            return Err(Refusal(
                "an owner-routed KV write reads its norm weights, RoPE positions and table",
            ));
        };
        q_norm.covers(dim * 2)?;
        k_norm.covers(dim * 2)?;
        rope_positions.covers(rows * 12)?;
        check_table(cos_sin, counts, g.rotary)?;
        Ok(vec![launch(
            self.entry.symbol(),
            counts.rows,
            self.entry.plan(counts.rows)?,
            pack(
                &self.entry.params(),
                vec![
                    buffer(buffers.qkv),
                    buffer(q_norm),
                    buffer(k_norm),
                    buffer(rope_positions),
                    buffer(cos_sin),
                    buffer(buffers.token_batch),
                    buffer(buffers.positions),
                    buffer(buffers.live_rows),
                    buffer(buffers.block_tables),
                    buffer(cache.k),
                    buffer(cache.v),
                    buffer(buffers.q),
                    Argument::U32(self.max_pages),
                    Argument::U64(cache.k_slot_stride),
                    Argument::U64(cache.v_slot_stride),
                    Argument::U32(g.eps.to_bits()),
                    scale_bits(cache.k_scale),
                    scale_bits(cache.v_scale),
                ],
            )?,
        )])
    }
}

/// `owned_index_key_write`: the indexer key write's warp per row, storing
/// a block key only on the rank that owns its page.
struct OwnedKeyWrite {
    entry: owned_writes::Entry,
    capacity: u32,
    max_pages: u32,
    group_rank: u32,
    info: OpInfo,
}

impl PreparedOp<SparseKeyWriteBuffers<View>> for OwnedKeyWrite {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &SparseKeyWriteBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let g = self.entry.indexer;
        let (heads, dim) = (u64::from(g.index_heads), u64::from(g.index_dim));
        buffers
            .projection
            .covers(bytes(&[rows, heads + 1, dim, 2])?)?;
        buffers.weight.covers(dim * 2)?;
        buffers.rope_positions.covers(rows * 12)?;
        check_table(buffers.cos_sin, counts, g.rotary_dim)?;
        buffers.token_batch.covers(rows * 4)?;
        buffers.positions.covers(rows * 4)?;
        buffers.live_rows.covers(4)?;
        let items = items(counts);
        for per_item in [buffers.slot_ids, buffers.anchor_slots, buffers.anchor_ends] {
            per_item.covers(items * 4)?;
        }
        check_rings(&buffers.rings, g.ring, g.index_dim)?;
        check_rings(&buffers.anchors, g.ring, g.index_dim)?;
        buffers
            .block_tables
            .covers(bytes(&[items, u64::from(self.max_pages), 4])?)?;
        let block_keys = &buffers.block_keys;
        check_block_keys(block_keys, g.block_page_bytes())?;
        Ok(vec![launch(
            self.entry.symbol(),
            counts.rows,
            self.entry.plan(counts.rows)?,
            pack(
                &self.entry.params(),
                vec![
                    buffer(buffers.projection),
                    buffer(buffers.weight),
                    buffer(buffers.rope_positions),
                    buffer(buffers.cos_sin),
                    buffer(buffers.token_batch),
                    buffer(buffers.positions),
                    buffer(buffers.live_rows),
                    buffer(buffers.slot_ids),
                    buffer(buffers.rings.keys),
                    buffer(buffers.rings.positions),
                    buffer(buffers.anchor_slots),
                    buffer(buffers.anchor_ends),
                    buffer(buffers.anchors.keys),
                    buffer(buffers.anchors.positions),
                    buffer(buffers.block_tables),
                    buffer(block_keys.keys),
                    Argument::U32(self.max_pages),
                    Argument::U64(block_keys.page_stride),
                    Argument::U32(g.eps.to_bits()),
                    Argument::U32(self.group_rank),
                    Argument::U32(self.entry.ranks),
                ],
            )?,
        )])
    }
}

/// How a group transfer reaches its members.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Transport {
    /// The shared all-gather: one CTA per 4096-word tile, every source
    /// sending to every receiver at once.
    Shared,
    /// The phased group gather: one cluster, one destination at a time.
    Phased,
}

impl Transport {
    fn name(self) -> &'static str {
        match self {
            Transport::Shared => "allgather",
            Transport::Phased => "phased",
        }
    }
}

/// A group transfer over the members' views in group-rank order.
struct Transfer {
    op: GroupTransferOp,
    rank: GroupRank,
    transport: Transport,
    capacity: u32,
    plan: LaunchPlan,
    info: OpInfo,
}

impl PreparedOp<GroupTransferBuffers<View>> for Transfer {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &GroupTransferBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        rows(counts, self.capacity)?;
        let op = self.op;
        if buffers.ranks.len() != op.size as usize {
            return Err(Refusal("a group transfer takes the views of every member"));
        }
        let overflow = Refusal("group transfer bytes overflow u64");
        buffers.input.covers(op.input_bytes().ok_or(overflow)?)?;
        let output_bytes = op.output_bytes().ok_or(overflow)?;
        let mut states = Vec::with_capacity(buffers.ranks.len());
        for (member, views) in buffers.ranks.iter().enumerate() {
            let [flags, epoch] = views.state[..] else {
                return Err(Refusal("a group member lacks its state regions"));
            };
            flags.covers(self.info.state[0])?;
            epoch.covers(self.info.state[1])?;
            if op.receives(member as u32) {
                views.output.covers(output_bytes)?;
            }
            states.push(GatherState {
                output: views.output.handle,
                flags: flags.handle,
                epoch: epoch.handle,
            });
        }
        let input = buffers.input.handle;
        let arguments = match (self.transport, op.transfer) {
            (Transport::Shared, GroupTransfer::OwnerGather { owner }) => {
                group::owner_gather_arguments(self.rank, &states, input, op.words, owner)
            }
            (Transport::Shared, _) => group::gather_arguments(self.rank, &states, input, op.words),
            (Transport::Phased, GroupTransfer::Exchange) => {
                phased::exchange_arguments(self.rank, &states, input, op.words)
            }
            (Transport::Phased, _) => phased::gather_arguments(self.rank, &states, input, op.words),
        }?;
        Ok(vec![Launch {
            collective: true,
            ..launch(
                self.info.symbols[0].0.clone(),
                counts.rows,
                self.plan,
                arguments,
            )
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Sm120, view};
    use super::*;
    use crate::attention::{gated, sparse};
    use crate::{Profile, ProfileNames};
    use kernel_core::backend::Backend;
    use kernel_core::collective::RankViews;
    use kernel_core::contract::AttentionPhase;
    use kernel_core::gated_sparse::{BlockKeys, FoldInputs, KeyRings};

    const BIG: u64 = 1 << 40;
    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "context-parallel-backend-test",
        tag: "cp_backend_test",
    });

    /// A rank's latent heads merged over pooled key lists, and a GQA rank's
    /// query heads over positions of one entry.
    fn latent() -> cp::Geometry {
        cp::Geometry {
            heads: 16,
            width: 512,
            page_size: 64,
            list_width: 2051,
            capacity: 8192,
        }
    }

    fn gqa() -> cp::Geometry {
        cp::Geometry {
            heads: 6,
            width: 256,
            page_size: 64,
            list_width: 1,
            capacity: 8192,
        }
    }

    /// A launch's arguments bound by parameter name, in the entry's
    /// parameter order, as the models' lowerings bind them.
    fn by_name(params: &[EntryParam], values: &[(&str, Argument)]) -> ArgumentList {
        assert_eq!(params.len(), values.len());
        ArgumentList(
            params
                .iter()
                .map(|param| {
                    let name = match param {
                        EntryParam::DevicePointer { name, .. }
                        | EntryParam::U32 { name }
                        | EntryParam::U64 { name } => name,
                        other => panic!("unexpected parameter {other:?}"),
                    };
                    values
                        .iter()
                        .find(|(bound, _)| bound == name)
                        .unwrap_or_else(|| panic!("{name} is unbound"))
                        .1
                        .clone()
                })
                .collect(),
        )
    }

    fn handle(argument: u64) -> Argument {
        Argument::DeviceBuffer(argument)
    }

    fn merge_query(ranks: u32, g: cp::Geometry) -> ContextParallelQuery {
        ContextParallelQuery::PartialMerge {
            ranks,
            heads: g.heads,
            width: g.width,
            capacity: g.capacity,
        }
    }

    fn localize_query(ranks: u32, g: cp::Geometry) -> ContextParallelQuery {
        ContextParallelQuery::LocalizeKeys {
            ranks,
            page: g.page_size,
            width: g.list_width,
            capacity: g.capacity,
        }
    }

    fn merge_buffers(layout: MergeLayout) -> PartialMergeBuffers<View> {
        PartialMergeBuffers {
            partials: view(1, BIG),
            rank_rows: layout.rank_rows,
            input_heads: layout.input_heads,
            head_offset: layout.head_offset,
            live_rows: view(2, 4),
            output: view(3, BIG),
        }
    }

    fn localize_buffers() -> LocalizeKeysBuffers<View> {
        LocalizeKeysBuffers {
            lists: view(4, BIG),
            live_rows: view(5, 4),
            output: view(6, BIG),
        }
    }

    /// The entries a model registers: every entry of a pooled list
    /// geometry, and of a geometry of one-entry lists the merges and the
    /// positions' localizations (its key list localizations are the same
    /// kernel under another symbol, which would tie).
    fn registered(g: cp::Geometry) -> Vec<cp::Entry> {
        cp::entries(PROFILE, g)
            .into_iter()
            .filter(|entry| g.list_width > 1 || !entry.name.contains("_localize_keys"))
            .collect()
    }

    #[test]
    fn merges_and_localizations_launch_what_the_builders_plan() {
        for g in [latent(), gqa()] {
            let entries = registered(g);
            let backend = Sm120::new().with_context_parallel(entries.clone());
            for &ranks in &cp::GROUP_SIZES {
                let merge = backend
                    .partial_merge(&merge_query(ranks, g), Need::rows(g.capacity))
                    .unwrap();
                let legacy = entries
                    .iter()
                    .find(|entry| entry.kind == Kind::PartialMerge { ranks })
                    .unwrap();
                assert!(merge.info().implementation.contains(&legacy.symbol()));
                assert_eq!(merge.info().guarantees, Invariance::NONE);
                for rows in [1_u32, 24, 2048] {
                    let bucket = rows.next_power_of_two();
                    let layout = MergeLayout::exchanged(bucket, g.heads);
                    let launches = merge
                        .launches(&merge_buffers(layout), View::EMPTY, Counts::rows(rows))
                        .unwrap();
                    let expected = Launch {
                        symbol: legacy.symbol(),
                        rows,
                        plan: legacy.plan(rows).unwrap(),
                        arguments: by_name(
                            &legacy.params(),
                            &[
                                ("partials", handle(1)),
                                ("live_rows", handle(2)),
                                ("out", handle(3)),
                                ("rank_rows", Argument::U32(layout.rank_rows)),
                                ("input_heads", Argument::U32(layout.input_heads)),
                                ("head_offset", Argument::U32(layout.head_offset)),
                            ],
                        ),
                        collective: false,
                        persistent: false,
                    };
                    assert_eq!(launches, [expected]);
                }
                for group_rank in 0..ranks {
                    let localize = backend
                        .localize_keys(&localize_query(ranks, g), group_rank, Need::rows(256))
                        .unwrap();
                    let legacy = entries
                        .iter()
                        .find(|entry| {
                            entry.kind == Kind::LocalizeKeys { ranks }
                                && entry.geometry.list_width == g.list_width
                        })
                        .unwrap();
                    let launches = localize
                        .launches(&localize_buffers(), View::EMPTY, Counts::rows(200))
                        .unwrap();
                    assert_eq!(launches[0].symbol, legacy.symbol());
                    assert_eq!(launches[0].plan, legacy.plan(200).unwrap());
                    assert_eq!(
                        launches[0].arguments,
                        by_name(
                            &legacy.params(),
                            &[
                                ("lists", handle(4)),
                                ("live_rows", handle(5)),
                                ("out", handle(6)),
                                ("cp_rank", Argument::U32(group_rank)),
                            ],
                        )
                    );
                }
            }
        }
        // The positions' localization of a pooled list's geometry: the
        // width-1 entry, not the list's.
        let positions = Sm120::new()
            .with_context_parallel(cp::entries(PROFILE, latent()))
            .localize_keys(
                &localize_query(
                    4,
                    cp::Geometry {
                        list_width: 1,
                        ..latent()
                    },
                ),
                3,
                Need::rows(8),
            )
            .unwrap();
        assert!(
            positions
                .info()
                .implementation
                .contains("localize_positions4")
        );
    }

    #[test]
    fn merge_and_localization_choices_do_not_follow_the_registration_order() {
        let small = cp::Geometry {
            capacity: 256,
            ..gqa()
        };
        let entries: Vec<cp::Entry> = cp::entries(PROFILE, gqa())
            .into_iter()
            .chain(
                cp::entries(PROFILE, small)
                    .into_iter()
                    .map(|entry| cp::Entry {
                        name: format!("{}_small", entry.name),
                        ..entry
                    }),
            )
            .filter(|entry| entry.kind == Kind::PartialMerge { ranks: 2 })
            .collect();
        // The same entry registered twice is one entry.
        let entries = [entries.clone(), entries[..1].to_vec()].concat();
        for order in select::orders(&entries) {
            let backend = Sm120::new().with_context_parallel(order);
            let symbol = |rows| {
                backend
                    .partial_merge(&merge_query(2, gqa()), Need::rows(rows))
                    .map(|op| op.info().symbols[0].0.clone())
            };
            assert!(symbol(200).unwrap().ends_with("_partial_merge2_small_v1"));
            assert!(symbol(300).unwrap().ends_with("_partial_merge2_v1"));
            assert!(symbol(8193).is_err());
        }
        // Two entries of one capacity that launch differently tie.
        let tied = cp::entries(PROFILE, gqa())
            .into_iter()
            .filter(|entry| entry.kind == Kind::PartialMerge { ranks: 2 })
            .flat_map(|entry| {
                [
                    entry.clone(),
                    cp::Entry {
                        name: format!("{}_other", entry.name),
                        ..entry
                    },
                ]
            })
            .collect::<Vec<_>>();
        for order in select::orders(&tied) {
            assert!(matches!(
                Sm120::new()
                    .with_context_parallel(order)
                    .partial_merge(&merge_query(2, gqa()), Need::rows(8)),
                Err(Refused::Rule(rule)) if rule.contains("entries serve the op")
            ));
        }
        // A geometry of one-entry lists instantiates its key list and
        // positions' localizations as one kernel under two symbols, so a
        // model registers one of them.
        assert!(matches!(
            Sm120::new()
                .with_context_parallel(cp::entries(PROFILE, gqa()))
                .localize_keys(&localize_query(2, gqa()), 0, Need::rows(8)),
            Err(Refused::Rule(rule)) if rule.contains("entries serve the op")
        ));
    }

    #[test]
    fn merges_and_localizations_refuse_what_they_cannot_serve() {
        let backend = Sm120::new().with_context_parallel(registered(gqa()));
        let merge = merge_query(2, gqa());
        // An unmet need, a group size no entry serves, a capacity past the
        // entries', a rank outside the group and a query of another kind.
        assert!(matches!(
            backend.partial_merge(&merge, Need::rows(8).with_exact(Invariance::one_row(8))),
            Err(Refused::Inexact { .. })
        ));
        assert!(matches!(
            backend.partial_merge(&merge_query(8, gqa()), Need::rows(8)),
            Err(Refused::Rule(rule)) if rule.contains("no SM120 partial merge entry")
        ));
        assert!(backend.partial_merge(&merge, Need::rows(8193)).is_err());
        assert!(backend.partial_merge(&merge, Need::rows(0)).is_err());
        assert!(
            backend
                .localize_keys(&localize_query(2, gqa()), 2, Need::rows(8))
                .is_err()
        );
        assert!(backend.localize_keys(&merge, 0, Need::rows(8)).is_err());
        assert!(
            backend
                .partial_merge(&localize_query(2, gqa()), Need::rows(8))
                .is_err()
        );
        // Launches past the prepared rows, segments that do not hold the
        // rows' heads and views too small.
        let op = backend.partial_merge(&merge, Need::rows(32)).unwrap();
        let fits = MergeLayout::exchanged(32, 6);
        assert!(
            op.launches(&merge_buffers(fits), View::EMPTY, Counts::rows(33))
                .is_err()
        );
        assert!(
            op.launches(
                &merge_buffers(MergeLayout::exchanged(16, 6)),
                View::EMPTY,
                Counts::rows(32)
            )
            .is_err()
        );
        let segments = 2 * 32 * 6 * (256 + 2) * 4;
        let mut buffers = merge_buffers(fits);
        buffers.partials = view(1, segments);
        op.launches(&buffers, View::EMPTY, Counts::rows(32))
            .unwrap();
        buffers.partials = view(1, segments - 1);
        assert!(
            op.launches(&buffers, View::EMPTY, Counts::rows(32))
                .is_err()
        );
        // A segment layout whose bytes overflow u64 is refused, not wrapped
        // to a size a small view covers.
        let one = backend
            .partial_merge(
                &ContextParallelQuery::PartialMerge {
                    ranks: 2,
                    heads: 6,
                    width: 256,
                    capacity: 1,
                },
                Need::rows(1),
            )
            .unwrap();
        for partials in [4, BIG] {
            let mut buffers = merge_buffers(MergeLayout {
                rank_rows: 1 << 31,
                input_heads: 1 << 31,
                head_offset: 0,
            });
            buffers.partials = view(1, partials);
            assert_eq!(
                one.launches(&buffers, View::EMPTY, Counts::rows(1)),
                Err(Refusal("context-parallel bytes overflow"))
            );
        }
        let mut buffers = merge_buffers(fits);
        buffers.output = view(3, 32 * 6 * 256 * 2 - 1);
        assert!(
            op.launches(&buffers, View::EMPTY, Counts::rows(32))
                .is_err()
        );
        let localize = backend
            .localize_keys(&localize_query(2, gqa()), 1, Need::rows(32))
            .unwrap();
        let mut buffers = localize_buffers();
        buffers.output = view(6, 31 * 4);
        assert!(
            localize
                .launches(&buffers, View::EMPTY, Counts::rows(32))
                .is_err()
        );
    }

    /// The group size ceiling is SM120's: queries past it validate, and the
    /// builders and prepare refuse them, as before the limit moved.
    #[test]
    fn groups_past_the_comm_slots_are_refused_by_sm120() {
        let nine = merge_query(9, gqa());
        nine.validate().unwrap();
        assert_eq!(
            cp::check_query(&nine),
            Err(Refusal(
                "an SM120 context-parallel group holds at most 8 ranks"
            ))
        );
        cp::check_query(&merge_query(8, gqa())).unwrap();
        let entry = |kind| cp::Entry {
            profile: PROFILE,
            name: "cp_backend_test_nine".into(),
            geometry: gqa(),
            kind,
        };
        for kind in [
            Kind::PartialMerge { ranks: 9 },
            Kind::LocalizeKeys { ranks: 9 },
        ] {
            assert!(entry(kind).contract().is_err());
            assert!(entry(kind).plan(8).is_err());
        }
        let backend = Sm120::new().with_context_parallel([
            entry(Kind::PartialMerge { ranks: 9 }),
            entry(Kind::LocalizeKeys { ranks: 9 }),
        ]);
        for refused in [
            backend.partial_merge(&nine, Need::rows(8)).map(|_| ()),
            backend
                .localize_keys(&localize_query(9, gqa()), 0, Need::rows(8))
                .map(|_| ()),
        ] {
            assert!(matches!(refused, Err(Refused::Rule(rule)) if rule.contains("at most 8")));
        }
        for entry in owned_writes::entries(PROFILE, GATED, INDEXER, 9) {
            entry.query().validate().unwrap();
            assert!(entry.contract().is_err());
        }
        let backend =
            Sm120::new().with_owned_writes(owned_writes::entries(PROFILE, GATED, INDEXER, 9));
        let op = OwnedKvWriteOp {
            query: owned_writes::entries(PROFILE, GATED, INDEXER, 9)[0].query(),
            max_context: 262_144,
        };
        assert!(matches!(
            backend.owned_kv_write(&op, Need::rows(8)),
            Err(Refused::Rule(rule)) if rule.contains("at most 8")
        ));
        // The pooled indexer's top-k and block-id merges.
        use crate::pooled_indexer::{block_merge, head_sum};
        let selection = head_sum::Geometry {
            heads: 4,
            head_dim: 128,
            group: 4,
            page_size: 64,
            max_top_k: 2048,
            always_tail: false,
        };
        let merge = head_sum::Entry {
            profile: PROFILE,
            name: "cp_backend_test_head_sum_merge9".into(),
            geometry: selection,
            kind: head_sum::Kind::TopKMerge { ranks: 9 },
            capacity: 256,
        };
        assert!(merge.contract().is_err());
        for (ranks, refused) in [(8, false), (9, true)] {
            let blocks = block_merge::entries(PROFILE, selection, &[ranks]);
            assert_eq!(blocks[0].contract().is_err(), refused);
        }
    }

    /// A rank of six query heads over one KV head of 256, 64-token pages.
    const GQA_RANK: paged_partials::Geometry = paged_partials::Geometry {
        q_heads: 6,
        kv_heads: 1,
        head_dim: 256,
        page: 64,
    };

    fn cache(fp8: bool) -> KvCache<View> {
        KvCache {
            k: view(20, BIG),
            v: view(21, BIG),
            slots: 64,
            k_slot_stride: 64 * 256 * 2,
            v_slot_stride: 64 * 256 * 2,
            k_scale: if fp8 { 0.5 } else { 1.0 },
            v_scale: if fp8 { 0.25 } else { 1.0 },
        }
    }

    fn paged_op(kv: DType, phase: AttentionPhase, capacity: u32) -> PagedPartialsOp {
        PagedPartialsOp {
            query: ContextParallelQuery::PagedAttentionPartials {
                q_heads: 6,
                kv_heads: 1,
                head_dim: 256,
                page: 64,
                kv,
                phase,
                capacity,
            },
            softmax_scale: 0.0625,
            max_context: 262_144,
        }
    }

    fn paged_buffers(fp8: bool) -> PagedPartialsBuffers<View> {
        PagedPartialsBuffers {
            q: view(10, BIG),
            cache: cache(fp8),
            block_tables: view(11, BIG),
            token_batch: view(12, BIG),
            positions: view(13, BIG),
            live_rows: view(14, 4),
            max: view(15, BIG),
            den: view(16, BIG),
            num: view(17, BIG),
        }
    }

    #[test]
    fn paged_partials_launch_what_the_builder_plans() {
        let entries = paged_partials::entries(PROFILE, GQA_RANK);
        let backend = Sm120::new().with_paged_partials(entries.clone());
        for legacy in &entries {
            let fp8 = legacy.kv == DType::Fp8E4M3;
            let op = backend
                .paged_attention_partials(
                    &paged_op(legacy.kv, legacy.phase, legacy.capacity),
                    1,
                    2,
                    Need::rows(legacy.capacity),
                )
                .unwrap();
            let rows = legacy.capacity.min(200);
            let launches = op
                .launches(
                    &paged_buffers(fp8),
                    View::EMPTY,
                    Counts::rows(rows).with_items(8, 4096),
                )
                .unwrap();
            let expected = by_name(
                &legacy.params(),
                &[
                    ("q", handle(10)),
                    ("k_cache", handle(20)),
                    ("v_cache", handle(21)),
                    ("block_tables", handle(11)),
                    ("row_batch", handle(12)),
                    ("positions", handle(13)),
                    ("live_rows", handle(14)),
                    ("out_max", handle(15)),
                    ("out_den", handle(16)),
                    ("out_num", handle(17)),
                    ("max_pages", Argument::U32(4096)),
                    ("k_page_stride_bytes", Argument::U64(64 * 256 * 2)),
                    ("v_page_stride_bytes", Argument::U64(64 * 256 * 2)),
                    ("scale_bits", Argument::U32(0.0625_f32.to_bits())),
                    ("k_scale_bits", Argument::U32(cache(fp8).k_scale.to_bits())),
                    ("v_scale_bits", Argument::U32(cache(fp8).v_scale.to_bits())),
                    ("cp_rank", Argument::U32(1)),
                    ("cp_size", Argument::U32(2)),
                ],
            );
            assert_eq!(launches[0].symbol, legacy.symbol());
            assert_eq!(launches[0].plan, legacy.plan(rows).unwrap());
            assert_eq!(launches[0].arguments, expected);
        }
        // A group rank outside the group, a phase or head shape no entry
        // serves, an unmet need, and a cache too small.
        let decode = paged_op(DType::Bf16, AttentionPhase::Decode, 256);
        assert!(
            backend
                .paged_attention_partials(&decode, 2, 2, Need::rows(8))
                .is_err()
        );
        assert!(
            backend
                .paged_attention_partials(&decode, 0, 2, Need::rows(257))
                .is_err()
        );
        let mut wide = decode;
        wide.query = ContextParallelQuery::PagedAttentionPartials {
            q_heads: 6,
            kv_heads: 1,
            head_dim: 128,
            page: 64,
            kv: DType::Bf16,
            phase: AttentionPhase::Decode,
            capacity: 256,
        };
        assert!(
            backend
                .paged_attention_partials(&wide, 0, 2, Need::rows(8))
                .is_err()
        );
        assert!(matches!(
            backend.paged_attention_partials(
                &decode,
                0,
                2,
                Need::rows(8).with_exact(Invariance::one_row(8))
            ),
            Err(Refused::Inexact { .. })
        ));
        let op = backend
            .paged_attention_partials(&decode, 0, 2, Need::rows(8))
            .unwrap();
        let mut buffers = paged_buffers(false);
        buffers.cache.k = view(20, 63 * 64 * 256 * 2);
        assert!(op.launches(&buffers, View::EMPTY, Counts::rows(8)).is_err());
    }

    /// A rank of six query heads over one KV head of 256 and a four-head
    /// indexer of 128, blocks of 4, 64-token pages.
    const GATED: gated::Config = gated::Config {
        q_heads: 6,
        kv_heads: 1,
        head_dim: 256,
        rotary: 64,
        theta: 1e7,
        sections: [11, 11, 10],
        eps: 1e-6,
        page: 64,
        max_rows: 8192,
    };

    const INDEXER: sparse::Geometry = sparse::Geometry {
        index_heads: 4,
        index_dim: 128,
        rotary_dim: 64,
        theta: 1e7,
        mrope: [11, 11, 10],
        eps: 1e-6,
        block: 4,
        ring: 16,
        top_blocks: 512,
        max_context: 262_144,
        q_heads: 6,
        kv_heads: 1,
        head_dim: 256,
        page_size: 64,
        softmax_scale: 0.0625,
    };

    fn owned_entries() -> Vec<owned_writes::Entry> {
        owned_writes::entries(PROFILE, GATED, INDEXER, 2)
    }

    fn kv_write_buffers(fp8: bool) -> GatedKvWriteBuffers<View> {
        GatedKvWriteBuffers {
            qkv: view(30, BIG),
            q: view(31, BIG),
            cache: cache(fp8),
            block_tables: view(32, BIG),
            token_batch: view(33, BIG),
            positions: view(34, BIG),
            live_rows: view(35, 4),
            rope: FoldInputs {
                cos_sin: Some(view(36, BIG)),
                q_norm: Some(view(37, BIG)),
                k_norm: Some(view(38, BIG)),
                rope_positions: Some(view(39, BIG)),
            },
        }
    }

    fn key_write_buffers() -> SparseKeyWriteBuffers<View> {
        let rings = |keys, positions| KeyRings {
            keys: view(keys, BIG),
            positions: view(positions, BIG),
            slots: 16,
        };
        SparseKeyWriteBuffers {
            projection: view(40, BIG),
            weight: view(41, BIG),
            rope_positions: view(42, BIG),
            cos_sin: view(43, BIG),
            token_batch: view(44, BIG),
            positions: view(45, BIG),
            live_rows: view(46, 4),
            slot_ids: view(47, BIG),
            rings: rings(48, 49),
            anchor_slots: view(50, BIG),
            anchor_ends: view(51, BIG),
            anchors: rings(52, 53),
            block_tables: view(54, BIG),
            block_keys: BlockKeys {
                keys: view(55, BIG),
                slots: 64,
                page_stride: 4096,
            },
        }
    }

    #[test]
    fn owned_writes_launch_what_the_builders_plan() {
        let entries = owned_entries();
        let backend = Sm120::new().with_owned_writes(entries.clone());
        let counts = |rows| Counts::rows(rows).with_items(8, 0).with_positions(4096);
        for legacy in &entries[..2] {
            let owned_writes::Kind::KvWrite { kv } = legacy.kind else {
                unreachable!("the first entries write K and V");
            };
            let fp8 = kv == DType::Fp8E4M3;
            let op = OwnedKvWriteOp {
                query: legacy.query(),
                max_context: 262_144,
            };
            let prepared = backend.owned_kv_write(&op, Need::rows(8192)).unwrap();
            let launches = prepared
                .launches(&kv_write_buffers(fp8), View::EMPTY, counts(300))
                .unwrap();
            let expected = by_name(
                &legacy.params(),
                &[
                    ("qkv", handle(30)),
                    ("q_norm", handle(37)),
                    ("k_norm", handle(38)),
                    ("rope_positions", handle(39)),
                    ("cos_sin", handle(36)),
                    ("row_batch", handle(33)),
                    ("local_positions", handle(34)),
                    ("live_rows", handle(35)),
                    ("block_tables", handle(32)),
                    ("k_cache", handle(20)),
                    ("v_cache", handle(21)),
                    ("q", handle(31)),
                    ("max_pages", Argument::U32(4096)),
                    ("k_page_stride_bytes", Argument::U64(64 * 256 * 2)),
                    ("v_page_stride_bytes", Argument::U64(64 * 256 * 2)),
                    ("eps_bits", Argument::U32(1e-6_f32.to_bits())),
                    ("k_scale_bits", Argument::U32(cache(fp8).k_scale.to_bits())),
                    ("v_scale_bits", Argument::U32(cache(fp8).v_scale.to_bits())),
                ],
            );
            assert_eq!(launches[0].symbol, legacy.symbol());
            assert_eq!(launches[0].plan, legacy.plan(300).unwrap());
            assert_eq!(launches[0].arguments, expected);
        }
        let legacy = &entries[2];
        let op = OwnedKeyWriteOp {
            query: legacy.query(),
            index_heads: 4,
            max_context: 262_144,
        };
        for group_rank in [0, 1] {
            let prepared = backend
                .owned_key_write(&op, group_rank, Need::rows(8192))
                .unwrap();
            let launches = prepared
                .launches(&key_write_buffers(), View::EMPTY, counts(301))
                .unwrap();
            let expected = by_name(
                &legacy.params(),
                &[
                    ("projection", handle(40)),
                    ("weight", handle(41)),
                    ("rope_positions", handle(42)),
                    ("cos_sin", handle(43)),
                    ("row_batch", handle(44)),
                    ("positions", handle(45)),
                    ("live_rows", handle(46)),
                    ("slot_ids", handle(47)),
                    ("ring_keys", handle(48)),
                    ("ring_positions", handle(49)),
                    ("anchor_slots", handle(50)),
                    ("anchor_ends", handle(51)),
                    ("anchor_keys", handle(52)),
                    ("anchor_positions", handle(53)),
                    ("block_tables", handle(54)),
                    ("block_keys", handle(55)),
                    ("max_pages", Argument::U32(4096)),
                    ("block_page_stride_bytes", Argument::U64(4096)),
                    ("eps_bits", Argument::U32(1e-6_f32.to_bits())),
                    ("cp_rank", Argument::U32(group_rank)),
                    ("cp_size", Argument::U32(2)),
                ],
            );
            assert_eq!(launches[0].symbol, legacy.symbol());
            assert_eq!(launches[0].plan, legacy.plan(301).unwrap());
            assert_eq!(launches[0].arguments, expected);
        }
        // A rank outside the group, another group size, other index heads,
        // an unmet need, and a missing norm weight.
        assert!(backend.owned_key_write(&op, 2, Need::rows(8)).is_err());
        let mut other = op;
        other.index_heads = 8;
        assert!(backend.owned_key_write(&other, 0, Need::rows(8)).is_err());
        let wider = OwnedKvWriteOp {
            query: owned_writes::entries(PROFILE, GATED, INDEXER, 4)[0].query(),
            max_context: 262_144,
        };
        assert!(matches!(
            backend.owned_kv_write(&wider, Need::rows(8)),
            Err(Refused::Rule(rule)) if rule.contains("no SM120")
        ));
        assert!(matches!(
            backend.owned_key_write(&op, 0, Need::rows(8).with_exact(Invariance::one_row(8))),
            Err(Refused::Inexact { .. })
        ));
        let kv = backend
            .owned_kv_write(
                &OwnedKvWriteOp {
                    query: entries[0].query(),
                    max_context: 262_144,
                },
                Need::rows(8),
            )
            .unwrap();
        let mut buffers = kv_write_buffers(false);
        buffers.rope.q_norm = None;
        assert!(kv.launches(&buffers, View::EMPTY, counts(8)).is_err());
        assert!(
            kv.launches(&kv_write_buffers(false), View::EMPTY, counts(9))
                .is_err()
        );
    }

    /// Each paged partials and owner-routed write op takes its entry in
    /// every registration order, a repeated registration counting once.
    #[test]
    fn partials_and_write_choices_do_not_follow_the_registration_order() {
        let paged = paged_partials::entries(PROFILE, GQA_RANK);
        let mut picked = Vec::new();
        for order in select::orders(&paged[..4]) {
            let order = [order.clone(), order[..1].to_vec()].concat();
            let backend = Sm120::new().with_paged_partials(order);
            let symbols = paged[..4]
                .iter()
                .map(|entry| {
                    backend
                        .paged_attention_partials(
                            &paged_op(entry.kv, entry.phase, entry.capacity),
                            0,
                            2,
                            Need::rows(8),
                        )
                        .unwrap()
                        .info()
                        .symbols[0]
                        .0
                        .clone()
                })
                .collect::<Vec<_>>();
            picked.push(symbols);
        }
        assert!(picked.iter().all(|symbols| *symbols == picked[0]));
        assert_eq!(
            picked[0],
            paged[..4]
                .iter()
                .map(paged_partials::Entry::symbol)
                .collect::<Vec<_>>()
        );
        let owned = owned_entries();
        for order in select::orders(&[owned.clone(), owned[..1].to_vec()].concat()) {
            let backend = Sm120::new().with_owned_writes(order);
            for entry in &owned[..2] {
                let op = OwnedKvWriteOp {
                    query: entry.query(),
                    max_context: 262_144,
                };
                assert_eq!(
                    backend
                        .owned_kv_write(&op, Need::rows(8))
                        .unwrap()
                        .info()
                        .symbols[0]
                        .0,
                    entry.symbol()
                );
            }
            let op = OwnedKeyWriteOp {
                query: owned[2].query(),
                index_heads: 4,
                max_context: 262_144,
            };
            assert_eq!(
                backend
                    .owned_key_write(&op, 1, Need::rows(8))
                    .unwrap()
                    .info()
                    .symbols[0]
                    .0,
                owned[2].symbol()
            );
        }
    }

    fn transfer_buffers(size: u32, state: [u64; 2]) -> GroupTransferBuffers<View> {
        GroupTransferBuffers {
            input: view(0xbeef, BIG),
            ranks: (0..u64::from(size))
                .map(|member| RankViews {
                    output: view(0x1000 * (member + 1), BIG),
                    state: vec![
                        view(0x10 * (member + 1), state[0]),
                        view(member + 100, state[1]),
                    ],
                })
                .collect(),
        }
    }

    fn states(size: u32) -> Vec<GatherState> {
        (0..u64::from(size))
            .map(|member| GatherState {
                output: 0x1000 * (member + 1),
                flags: 0x10 * (member + 1),
                epoch: member + 100,
            })
            .collect()
    }

    #[test]
    fn group_transfers_launch_what_the_group_builders_give() {
        let transfer = |transfer, size, words| GroupTransferOp {
            transfer,
            size,
            words,
        };
        let shared = Sm120::new();
        let phased_backend = Sm120::new().with_phased_group_gather();
        for size in cp::GROUP_SIZES {
            for group_rank in 0..size {
                let rank = GroupRank::of(group_rank, size, size).unwrap();
                let words = 32 * 16 * 256;
                let prepared = |backend: &Sm120, op| {
                    let op = backend
                        .group_transfer(&op, group_rank, Need::rows(32))
                        .unwrap();
                    let state = [op.info().state[0], op.info().state[1]];
                    let launches = op
                        .launches(
                            &transfer_buffers(size, state),
                            View::EMPTY,
                            Counts::rows(24),
                        )
                        .unwrap();
                    assert_eq!(launches.len(), 1);
                    assert!(launches[0].collective && !launches[0].persistent);
                    assert_eq!(launches[0].rows, 24);
                    (op.info().clone(), launches[0].clone())
                };
                // Gathers and barriers through the shared all-gather.
                let gathers = group::gather_buffers(size, words).unwrap();
                for kind in [GroupTransfer::Gather, GroupTransfer::Barrier] {
                    let (info, launch) = prepared(&shared, transfer(kind, size, words));
                    assert_eq!(info.state, [gathers.flag_bytes, gathers.epoch_bytes]);
                    assert_eq!(launch.symbol, group::ALLGATHER_SYMBOL);
                    assert_eq!(launch.plan, group::gather_plan(words).unwrap());
                    assert_eq!(
                        launch.arguments,
                        group::gather_arguments(rank, &states(size), 0xbeef, words).unwrap()
                    );
                }
                // The barrier stays there under the phased gather.
                let (_, barrier) =
                    prepared(&phased_backend, transfer(GroupTransfer::Barrier, size, 1));
                assert_eq!(barrier.symbol, group::ALLGATHER_SYMBOL);
                // Owner gathers on either backend.
                for owner in 0..size {
                    for backend in [&shared, &phased_backend] {
                        let (_, launch) = prepared(
                            backend,
                            transfer(GroupTransfer::OwnerGather { owner }, size, words),
                        );
                        assert_eq!(
                            launch.arguments,
                            group::owner_gather_arguments(
                                rank,
                                &states(size),
                                0xbeef,
                                words,
                                owner
                            )
                            .unwrap()
                        );
                    }
                }
                // Gathers and exchanges through the phased gather.
                for (kind, arguments) in [
                    (
                        GroupTransfer::Gather,
                        phased::gather_arguments(rank, &states(size), 0xbeef, words),
                    ),
                    (
                        GroupTransfer::Exchange,
                        phased::exchange_arguments(rank, &states(size), 0xbeef, words),
                    ),
                ] {
                    let (info, launch) = prepared(&phased_backend, transfer(kind, size, words));
                    assert_eq!(info.state, [phased::flag_bytes(size), phased::EPOCH_BYTES]);
                    assert_eq!(launch.symbol, phased::SYMBOL);
                    assert_eq!(launch.plan, phased::plan());
                    assert_eq!(launch.arguments, arguments.unwrap());
                }
            }
        }
    }

    /// A transfer reports the comm declaration of the entry its transport
    /// launches: the shared all-gather's, and the phased gather's, both
    /// moves of up to 8192 rows.
    #[test]
    fn group_transfers_claim_their_transport_declaration() {
        let declared = |symbol: &str| {
            crate::comm::declared_entries()
                .into_iter()
                .find(|declared| declared.symbol == symbol)
                .map_or(Invariance::NONE, |declared| declared.invariance)
        };
        let op = |transfer| GroupTransferOp {
            transfer,
            size: 2,
            words: 1024,
        };
        for transfer in [
            GroupTransfer::Gather,
            GroupTransfer::OwnerGather { owner: 1 },
            GroupTransfer::Barrier,
        ] {
            let shared = Sm120::new()
                .group_transfer(&op(transfer), 0, Need::rows(8))
                .unwrap();
            assert_eq!(
                shared.info().guarantees,
                declared(group::ALLGATHER_SYMBOL),
                "{transfer:?}"
            );
        }
        let phased = Sm120::new()
            .with_phased_group_gather()
            .group_transfer(&op(GroupTransfer::Gather), 0, Need::rows(8))
            .unwrap();
        assert_eq!(phased.info().symbols[0].0, phased::SYMBOL);
        assert_eq!(phased.info().guarantees, declared(phased::SYMBOL));
        assert_eq!(phased.info().guarantees, Invariance::any_rows(8192));
        assert!(
            Sm120::new()
                .with_phased_group_gather()
                .group_transfer(
                    &op(GroupTransfer::Gather),
                    0,
                    Need::rows(8).with_exact(Invariance::one_row(4))
                )
                .is_ok()
        );
        assert!(matches!(
            Sm120::new().with_phased_group_gather().group_transfer(
                &op(GroupTransfer::Gather),
                0,
                Need::rows(8).with_exact(Invariance::any_rows(16384))
            ),
            Err(Refused::Inexact { .. })
        ));
    }

    /// The shared all-gather is declared a move of up to 8192 rows, so the
    /// shared transfers grant a bucket's exact verify and prefill chunk
    /// needs, and refuse rows past the declaration.
    #[test]
    fn shared_group_transfers_grant_the_declared_move() {
        let moved = Invariance::any_rows(8192);
        let needed = Invariance {
            chunk: kernel_core::properties::Chunk::Granule(64),
            rows: kernel_core::properties::Rows::OneRow { max: 4 },
            ..Invariance::NONE
        };
        let op = |transfer| GroupTransferOp {
            transfer,
            size: 2,
            words: 1024,
        };
        for transfer in [
            GroupTransfer::Gather,
            GroupTransfer::OwnerGather { owner: 1 },
            GroupTransfer::Barrier,
        ] {
            let shared = Sm120::new()
                .group_transfer(&op(transfer), 0, Need::rows(8).with_exact(needed))
                .unwrap();
            assert_eq!(shared.info().guarantees, moved, "{transfer:?}");
            assert!(matches!(
                Sm120::new().group_transfer(
                    &op(transfer),
                    0,
                    Need::rows(8).with_exact(Invariance::any_rows(16384))
                ),
                Err(Refused::Inexact { .. })
            ));
        }
    }

    /// Without a registered claim, the owned key write (which keeps key
    /// rings) grants no anchor need, at any span, with or without a chunk
    /// or rows claim; the owned K/V write (paged KV only) writes no
    /// per-slot state and meets anchor needs trivially.
    #[test]
    fn undeclared_owned_key_writes_grant_no_anchors() {
        use kernel_core::properties::{Anchors, Chunk};
        let owned = owned_entries();
        let key_write = owned
            .iter()
            .find(|entry| entry.kind == owned_writes::Kind::KeyWrite)
            .unwrap();
        let backend = Sm120::new().with_owned_writes(owned.clone());
        let op = OwnedKeyWriteOp {
            query: key_write.query(),
            index_heads: key_write.indexer.index_heads,
            max_context: key_write.indexer.max_context,
        };
        let plain = backend.owned_key_write(&op, 1, Need::rows(4)).unwrap();
        assert_eq!(plain.info().guarantees, STATEFUL_NONE);
        for anchors in [
            Anchors::Spare,
            Anchors::SpareAfter { rows: 4 },
            Anchors::SpareAfter { rows: 64 },
        ] {
            for chunk in [Chunk::None, Chunk::Granule(64)] {
                let need = Need::rows(4).with_exact(Invariance {
                    chunk,
                    anchors,
                    ..Invariance::NONE
                });
                assert!(matches!(
                    backend.owned_key_write(&op, 1, need),
                    Err(Refused::Inexact { .. })
                ));
            }
        }
        let kv_write = owned
            .iter()
            .find(|entry| matches!(entry.kind, owned_writes::Kind::KvWrite { .. }))
            .unwrap();
        let kv_op = OwnedKvWriteOp {
            query: kv_write.query(),
            max_context: 262_144,
        };
        let spare = Need::rows(4).with_exact(Invariance::NONE.with_anchors(Anchors::Spare));
        let kv = backend.owned_kv_write(&kv_op, spare).unwrap();
        assert_eq!(kv.info().guarantees, Invariance::NONE);
    }

    /// An op guarantees what the model declares for the entry it launches,
    /// and nothing for an entry it declares nothing for; a need the claim
    /// does not cover is refused.
    #[test]
    fn ops_report_the_claims_the_model_declares() {
        let claim = Invariance {
            chunk: kernel_core::properties::Chunk::Any,
            rows: kernel_core::properties::Rows::OneRow { max: 4 },
            ..Invariance::NONE
        };
        let entries = registered(gqa());
        let merge_symbol = entries
            .iter()
            .find(|entry| entry.kind == Kind::PartialMerge { ranks: 2 })
            .unwrap()
            .symbol();
        let owned = owned_entries();
        let key_write = owned
            .iter()
            .find(|entry| entry.kind == owned_writes::Kind::KeyWrite)
            .unwrap();
        let backend = Sm120::new()
            .with_context_parallel(entries.clone())
            .with_owned_writes(owned.clone())
            .with_context_parallel_claims([
                (merge_symbol.clone(), claim),
                (key_write.symbol(), claim),
                (merge_symbol.clone(), claim),
            ]);
        let merge = backend
            .partial_merge(&merge_query(2, gqa()), Need::rows(4).with_exact(claim))
            .unwrap();
        assert_eq!(merge.info().guarantees, claim);
        assert!(matches!(
            backend.partial_merge(
                &merge_query(2, gqa()),
                Need::rows(8).with_exact(Invariance::one_row(8))
            ),
            Err(Refused::Inexact { .. })
        ));
        // The localization of the same registration declares nothing.
        let localize = backend
            .localize_keys(&localize_query(2, gqa()), 1, Need::rows(4))
            .unwrap();
        assert_eq!(localize.info().guarantees, Invariance::NONE);
        assert!(matches!(
            backend.localize_keys(
                &localize_query(2, gqa()),
                1,
                Need::rows(4).with_exact(claim)
            ),
            Err(Refused::Inexact { .. })
        ));
        let key_op = OwnedKeyWriteOp {
            query: key_write.query(),
            index_heads: key_write.indexer.index_heads,
            max_context: key_write.indexer.max_context,
        };
        let prepared = backend
            .owned_key_write(&key_op, 1, Need::rows(4).with_exact(claim))
            .unwrap();
        assert_eq!(prepared.info().guarantees, claim);
        // Claims do not change what an op launches or its implementation.
        let unclaimed = Sm120::new().with_context_parallel(entries);
        let plain = unclaimed
            .partial_merge(&merge_query(2, gqa()), Need::rows(4))
            .unwrap();
        assert_eq!(plain.info().implementation, merge.info().implementation);
        let layout = MergeLayout::exchanged(4, 6);
        assert_eq!(
            plain
                .launches(&merge_buffers(layout), View::EMPTY, Counts::rows(3))
                .unwrap(),
            merge
                .launches(&merge_buffers(layout), View::EMPTY, Counts::rows(3))
                .unwrap()
        );
    }

    #[test]
    #[should_panic(expected = "claimed twice differently")]
    fn a_symbol_claimed_twice_differently_panics() {
        let _ = Sm120::new().with_context_parallel_claims([
            ("symbol".to_owned(), Invariance::one_row(4)),
            ("symbol".to_owned(), Invariance::one_row(2)),
        ]);
    }

    #[test]
    fn group_transfers_refuse_what_they_cannot_serve() {
        let op = |transfer, size| GroupTransferOp {
            transfer,
            size,
            words: 1024,
        };
        let shared = Sm120::new();
        // An exchange needs the phased gather; a rank outside the group, a
        // group past the comm slots, an unmet need.
        assert!(matches!(
            shared.group_transfer(&op(GroupTransfer::Exchange, 2), 0, Need::rows(8)),
            Err(Refused::Rule(rule)) if rule.contains("phased")
        ));
        assert!(
            shared
                .group_transfer(&op(GroupTransfer::Gather, 2), 2, Need::rows(8))
                .is_err()
        );
        assert!(
            shared
                .group_transfer(&op(GroupTransfer::Gather, 9), 0, Need::rows(8))
                .is_err()
        );
        assert!(matches!(
            shared.group_transfer(
                &op(GroupTransfer::Gather, 2),
                0,
                Need::rows(8).with_exact(
                    Invariance::one_row(8).with_ties(kernel_core::properties::Ties::LowerIndex)
                )
            ),
            Err(Refused::Inexact { .. })
        ));
        // Launches past the rows, without every member, with a small state,
        // input or receiving output; an owner gather leaves the others'
        // outputs unchecked.
        let gather = shared
            .group_transfer(&op(GroupTransfer::Gather, 2), 1, Need::rows(8))
            .unwrap();
        let state = [gather.info().state[0], gather.info().state[1]];
        let launches = |op: &Prepared<GroupTransferBuffers<View>>, buffers, rows| {
            op.launches(&buffers, View::EMPTY, Counts::rows(rows))
        };
        assert!(launches(&gather, transfer_buffers(2, state), 9).is_err());
        assert!(launches(&gather, transfer_buffers(1, state), 8).is_err());
        assert!(launches(&gather, transfer_buffers(2, [state[0] - 4, state[1]]), 8).is_err());
        let mut buffers = transfer_buffers(2, state);
        buffers.input = view(0xbeef, 4095);
        assert!(launches(&gather, buffers, 8).is_err());
        let mut buffers = transfer_buffers(2, state);
        buffers.ranks[0].output = view(0x1000, 2 * 4096 - 1);
        assert!(launches(&gather, buffers.clone(), 8).is_err());
        let owner = shared
            .group_transfer(
                &op(GroupTransfer::OwnerGather { owner: 1 }, 2),
                1,
                Need::rows(8),
            )
            .unwrap();
        launches(&owner, buffers, 8).unwrap();
        // An exchange's input holds a block per member.
        let exchange = Sm120::new()
            .with_phased_group_gather()
            .group_transfer(&op(GroupTransfer::Exchange, 2), 0, Need::rows(8))
            .unwrap();
        let state = [exchange.info().state[0], exchange.info().state[1]];
        let mut buffers = transfer_buffers(2, state);
        buffers.input = view(0xbeef, 2 * 4096 - 1);
        assert!(launches(&exchange, buffers, 8).is_err());
    }

    const OTHER: Profile = Profile::synthetic(ProfileNames {
        name: "context-parallel-backend-other",
        tag: "cp_backend_other",
    });

    /// The merge and localization keys: the query holds the kind, ranks and
    /// geometry the kind reads; the symbol is a launch fact; the profile is
    /// not.
    #[test]
    fn merge_and_localization_keys_hold_what_they_launch() {
        use super::select::{Class, Field, field_names, keys_follow_launches};
        let entries = registered(gqa());
        for (kind, other_kind) in [
            (
                Kind::PartialMerge { ranks: 2 },
                Kind::PartialMerge { ranks: 4 },
            ),
            (
                Kind::LocalizeKeys { ranks: 2 },
                Kind::LocalizeKeys { ranks: 4 },
            ),
        ] {
            let base = entries.iter().find(|e| e.kind == kind).unwrap().clone();
            let names = field_names!(cp::Entry {
                profile,
                name,
                geometry,
                kind
            });
            let fields: Vec<Field<cp::Entry>> = vec![
                ("profile", Class::Inert, Box::new(|e| e.profile = OTHER)),
                (
                    "name",
                    Class::Launch,
                    Box::new(|e| e.name.push_str("_twin")),
                ),
                (
                    "geometry",
                    Class::Selects,
                    Box::new(|e| {
                        e.geometry.heads += 1;
                        e.geometry.list_width += 1;
                    }),
                ),
                (
                    "kind",
                    Class::Selects,
                    Box::new(move |e| e.kind = other_kind),
                ),
            ];
            keys_follow_launches(&base, names, fields, cp_launch, |e| {
                let backend = Sm120::new().with_context_parallel([e.clone()]);
                let code = format!("{:?}", e.entry_spec());
                let launches = match kind {
                    Kind::PartialMerge { .. } => {
                        let op = backend.partial_merge(&merge_query(2, gqa()), Need::rows(8))?;
                        let buffers = merge_buffers(MergeLayout::exchanged(8, 6));
                        [1, 8].map(|rows| {
                            op.launches(&buffers, View::EMPTY, Counts::rows(rows))
                                .unwrap()
                        })
                    }
                    Kind::LocalizeKeys { .. } => {
                        let op =
                            backend.localize_keys(&localize_query(2, gqa()), 1, Need::rows(8))?;
                        [1, 8].map(|rows| {
                            op.launches(&localize_buffers(), View::EMPTY, Counts::rows(rows))
                                .unwrap()
                        })
                    }
                };
                Ok((code, launches))
            });
        }
    }

    /// The paged partials' key: their geometry, KV and phase select the op;
    /// the symbol is a launch fact; the profile and the capacity, which the
    /// selection settles, are not.
    #[test]
    fn paged_partials_keys_hold_what_they_launch() {
        use super::select::{Class, Field, field_names, keys_follow_launches};
        let base = paged_partials::entries(PROFILE, GQA_RANK).remove(0);
        let names = field_names!(paged_partials::Entry {
            profile,
            name,
            geometry,
            kv,
            phase,
            capacity
        });
        let fields: Vec<Field<paged_partials::Entry>> = vec![
            ("profile", Class::Inert, Box::new(|e| e.profile = OTHER)),
            (
                "name",
                Class::Launch,
                Box::new(|e| e.name.push_str("_twin")),
            ),
            (
                "geometry",
                Class::Selects,
                Box::new(|e| e.geometry.head_dim /= 2),
            ),
            ("kv", Class::Selects, Box::new(|e| e.kv = DType::Fp8E4M3)),
            (
                "phase",
                Class::Selects,
                Box::new(|e| e.phase = AttentionPhase::Prefill),
            ),
            ("capacity", Class::Inert, Box::new(|e| e.capacity *= 2)),
        ];
        let op = paged_op(base.kv, base.phase, base.capacity);
        keys_follow_launches(&base, names, fields, paged_launch, |e| {
            let prepared = Sm120::new()
                .with_paged_partials([e.clone()])
                .paged_attention_partials(&op, 0, 2, Need::rows(8))?;
            let launches = [1, 8].map(|rows| {
                prepared
                    .launches(
                        &paged_buffers(false),
                        View::EMPTY,
                        Counts::rows(rows).with_items(rows, 4096),
                    )
                    .unwrap()
            });
            Ok((format!("{:?}", e.entry_spec()), launches))
        });
    }

    /// The owner-routed writes' keys: the query selects the op (a KV
    /// write's gated config, a key write's indexer geometry, the kind and
    /// the group size); the symbol and a key write's top-k, which its body
    /// names, are launch facts; the profile, and the config the other kind
    /// reads, are not.
    #[test]
    fn owned_write_keys_hold_what_they_launch() {
        use super::select::{Class, Field, field_names, keys_follow_launches};
        let entries = owned_entries();
        let names = || {
            field_names!(owned_writes::Entry {
                profile,
                name,
                kind,
                gated,
                indexer,
                ranks
            })
        };
        let counts = |rows| Counts::rows(rows).with_items(rows, 0).with_positions(4096);
        let fields: Vec<Field<owned_writes::Entry>> = vec![
            ("profile", Class::Inert, Box::new(|e| e.profile = OTHER)),
            (
                "name",
                Class::Launch,
                Box::new(|e| e.name.push_str("_twin")),
            ),
            (
                "kind",
                Class::Selects,
                Box::new(|e| e.kind = owned_writes::Kind::KvWrite { kv: DType::Fp8E4M3 }),
            ),
            (
                "gated",
                Class::Selects,
                Box::new(|e| e.gated.head_dim = 128),
            ),
            (
                "indexer",
                Class::Inert,
                Box::new(|e| e.indexer.top_blocks /= 2),
            ),
            ("ranks", Class::Selects, Box::new(|e| e.ranks = 4)),
        ];
        let kv_op = OwnedKvWriteOp {
            query: entries[0].query(),
            max_context: 262_144,
        };
        keys_follow_launches(&entries[0], names(), fields, owned_launch, |e| {
            let prepared = Sm120::new()
                .with_owned_writes([e.clone()])
                .owned_kv_write(&kv_op, Need::rows(8))?;
            let launches = [1, 8].map(|rows| {
                prepared
                    .launches(&kv_write_buffers(false), View::EMPTY, counts(rows))
                    .unwrap()
            });
            Ok((format!("{:?}", e.entry_spec()), launches))
        });
        let fields: Vec<Field<owned_writes::Entry>> = vec![
            ("profile", Class::Inert, Box::new(|e| e.profile = OTHER)),
            (
                "name",
                Class::Launch,
                Box::new(|e| e.name.push_str("_twin")),
            ),
            (
                "kind",
                Class::Selects,
                Box::new(|e| e.kind = owned_writes::Kind::KvWrite { kv: DType::Bf16 }),
            ),
            ("gated", Class::Inert, Box::new(|e| e.gated.head_dim = 128)),
            (
                "indexer",
                Class::Launch,
                Box::new(|e| e.indexer.top_blocks /= 2),
            ),
            ("ranks", Class::Selects, Box::new(|e| e.ranks = 4)),
        ];
        let key_op = OwnedKeyWriteOp {
            query: entries[2].query(),
            index_heads: 4,
            max_context: 262_144,
        };
        keys_follow_launches(&entries[2], names(), fields, owned_launch, |e| {
            let prepared = Sm120::new()
                .with_owned_writes([e.clone()])
                .owned_key_write(&key_op, 1, Need::rows(8))?;
            let launches = [1, 8].map(|rows| {
                prepared
                    .launches(&key_write_buffers(), View::EMPTY, counts(rows))
                    .unwrap()
            });
            Ok((format!("{:?}", e.entry_spec()), launches))
        });
    }
}
