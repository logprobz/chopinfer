//! The pooled indexer through the entries a model instantiates:
//! `pooled_indexer` (key writes, weighted selections, their key list merges
//! and the compaction), `pooled_indexer::head_sum` (head-sum selections and
//! their key list merges) and `pooled_indexer::block_merge` (block-id
//! merges). An op is served by the selection's entry whose query is the
//! op's, apart from `capacity`, with a capacity that holds the model's.
//!
//! Each prepared op plans its entry at every row count up to the needed
//! capacity when it is prepared, as the entry's planner plans a launch of
//! that many rows: grids are sized for the step's rows, CTAs past the
//! device live count return, and a selection splits each row's pools by
//! the step's rows (`pooled_indexer::Entry::splits`). The selection is
//! exact, so the split count changes no output bit. No entry declares an
//! exactness property in its catalog (`crate::invariance`), so the ops
//! guarantee none.
//!
//! A weighted selection runs the forms the model registers for its entry
//! where they serve the row count ([`writer`]): at one split, the one-split
//! form (`pooled_indexer::rows`) in place of the split kernel and its
//! merge; at two or more, the radix merge (`merge_radix`, global
//! selections only) or else the tree merge (`merge_tree`) in place of the
//! split merge. Every form writes the entry's key lists or pairs bit for
//! bit (README "One-split selection", "Tree merge", "Radix merge"), so the
//! op keeps the entry's implementation name; the tree and the radix merge
//! give equal bits by construction, and the radix merge is taken where
//! both are registered. Without forms the entry runs itself at every row
//! count.
//!
//! The ops run at context-parallel group rank 0 of 1, except a `PerRank`
//! selection, which runs at its group rank of its rank count, and a key
//! write, which runs at the group rank and size it is prepared for and
//! stores the pools of that rank's own pages
//! (`pooled_indexer::pool_address`). Context parallelism adds the group
//! all-gather of the per-rank pairs between the selection and the merge,
//! and the localization of the merged key lists; those belong to the
//! context-parallel family.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::contract::{
    Argument, ArgumentList, DType, LaunchPlan, RecurrentPhase, Refusal, StaticQuery, Validate,
};
use kernel_core::pooled_indexer::{
    IndexerKeyWriteBuffers, IndexerMergeBuffers, IndexerSelectBuffers, IndexerSelectionMode,
    PooledIndexerQuery, PooledKeyPages, SelectionCompactBuffers,
};
use kernel_core::properties::Invariance;

use super::check_plan;
use super::context_parallel::Claims;
use super::select::{Candidate, choose};
use crate::pooled_indexer::{
    Entry, Kind, TILE_POOLS, block_merge, head_sum, merge_radix, merge_tree, rows,
};

/// A model's pooled indexer entries.
#[derive(Default)]
pub(super) struct Selection {
    pub weighted: Vec<Entry>,
    pub head_sum: Vec<head_sum::Entry>,
    pub block_merges: Vec<block_merge::Entry>,
    /// One-split forms of weighted selection entries.
    pub one_split: Vec<rows::Entry>,
    /// Tree merges of weighted selection entries' split lists.
    pub trees: Vec<merge_tree::Entry>,
    /// Radix merges of global weighted selection entries' split lists.
    pub radixes: Vec<merge_radix::Entry>,
}

const BF16: u64 = 2;
const I32: u64 = 4;
/// A per-rank pair: U32 score bits and U32 pool.
const PAIR: u64 = 8;

/// `query` with its `capacity` cleared: the field a backend sizes for.
fn uncapped(query: PooledIndexerQuery) -> PooledIndexerQuery {
    use PooledIndexerQuery::*;
    match query {
        KeyWrite {
            pooling,
            key_norm_eps_bits,
            page_size,
            key_format,
            phase,
            ..
        } => KeyWrite {
            pooling,
            key_norm_eps_bits,
            page_size,
            key_format,
            phase,
            capacity: 0,
        },
        Select {
            heads,
            head_dim,
            group,
            max_top_k,
            always_tail,
            granularity,
            key_format,
            mode,
            page_size,
            ..
        } => Select {
            heads,
            head_dim,
            group,
            max_top_k,
            always_tail,
            granularity,
            key_format,
            mode,
            page_size,
            capacity: 0,
        },
        HeadSumSelect {
            heads,
            head_dim,
            group,
            max_top_k,
            always_tail,
            mode,
            page_size,
            ..
        } => HeadSumSelect {
            heads,
            head_dim,
            group,
            max_top_k,
            always_tail,
            mode,
            page_size,
            capacity: 0,
        },
        CompactSelection { width, .. } => CompactSelection { width, capacity: 0 },
    }
}

/// A merge `query` with its `capacity` cleared; other queries unchanged.
fn uncapped_merge(query: ContextParallelQuery) -> ContextParallelQuery {
    use ContextParallelQuery::*;
    match query {
        TopKMerge {
            ranks,
            group,
            max_top_k,
            always_tail,
            ..
        } => TopKMerge {
            ranks,
            group,
            max_top_k,
            always_tail,
            capacity: 0,
        },
        TopKMergeBlocks {
            ranks,
            group,
            max_top_k,
            ..
        } => TopKMergeBlocks {
            ranks,
            group,
            max_top_k,
            capacity: 0,
        },
        other => other,
    }
}

/// Whether static query `query` is `op` apart from capacity.
fn serves(query: StaticQuery, op: PooledIndexerQuery) -> bool {
    matches!(query, StaticQuery::PooledIndexer(q) if uncapped(q) == uncapped(op))
}

/// Whether static query `query` is merge `op` apart from capacity.
fn serves_merge(query: StaticQuery, op: ContextParallelQuery) -> bool {
    matches!(query, StaticQuery::ContextParallel(q) if uncapped_merge(q) == uncapped_merge(op))
}

/// The shapes the SM120 templates instantiate: 128-wide keys in pools of
/// 2, 4 or 8 tokens; weighted selections over 16 to 64 heads in steps of
/// 16, head-sum selections over four; a kept pool count that is a power of
/// two of at least one scoring tile; verify key writes of at most 16 rows
/// per item; per-rank selections over at most `MAX_COMM_RANKS` ranks.
fn limits(op: &PooledIndexerQuery) -> Result<(), Refused> {
    crate::pooled_indexer::check_ranks(op)?;
    let kept = |group: u32, max_top_k: u32| {
        let kept = max_top_k / group;
        kept.is_power_of_two() && kept >= TILE_POOLS
    };
    let pool = |group: u32| matches!(group, 2 | 4 | 8);
    let inside = match *op {
        PooledIndexerQuery::KeyWrite { pooling, phase, .. } => {
            pooling.width == 128
                && pool(pooling.group)
                && !matches!(phase, RecurrentPhase::Verify { max_rows } if max_rows > 16)
        }
        PooledIndexerQuery::Select {
            heads,
            head_dim,
            group,
            max_top_k,
            ..
        } => {
            heads.is_multiple_of(16)
                && heads <= 64
                && head_dim == 128
                && pool(group)
                && kept(group, max_top_k)
        }
        PooledIndexerQuery::HeadSumSelect {
            heads,
            head_dim,
            group,
            max_top_k,
            ..
        } => heads == 4 && head_dim == 128 && pool(group) && kept(group, max_top_k),
        PooledIndexerQuery::CompactSelection { .. } => true,
    };
    if inside {
        Ok(())
    } else {
        Err(Refused::Rule(
            "pooled indexer geometry is outside the SM120 engine".into(),
        ))
    }
}

/// The entry of `found` (each with its capacity, symbol and launch key)
/// with the least capacity that holds the model's rows, whatever the order
/// the selection registered them in (`select::choose`), and its
/// guarantees. The entries carry no claims: each guarantees what
/// `guarantees` gives it, the model's declared claims for a
/// context-parallel entry (`Claims`) and nothing otherwise.
fn fitting<E: std::fmt::Debug>(
    found: impl Iterator<Item = (E, u32, String, String)>,
    guarantees: impl Fn(&E) -> Invariance,
    need: Need,
) -> Result<(E, Invariance), Refused> {
    choose(
        "pooled indexer",
        found.map(|(entry, capacity, symbol, launches)| {
            let claimed = guarantees(&entry);
            Candidate::new(entry, symbol, launches, claimed).capacity(capacity)
        }),
        need,
    )
    .map(|chosen| (chosen.entry, chosen.guarantees))
}

/// What a weighted entry launches: its symbol and kind, and the geometry
/// its kind's kernel reads (`cuda/pooled_indexer`). A key write reads its
/// key width, pools, pages, position type and norm (`KeyWrite`); a
/// selection its heads, pools, pages, top-k and tail (`Select`); the merge
/// its pools, top-k and tail (`topk_merge`), never the heads or pages its
/// traits also name; a compaction its list width.
fn weighted_launch(entry: &Entry) -> String {
    let Entry {
        profile: _,  // filters catalogs; no launch reads it
        name: _,     // in the symbol
        geometry,    // by kind, below
        kind,        // selects the op
        capacity: _, // bounds the rows the selection settles
    } = entry;
    let crate::pooled_indexer::Geometry {
        heads,
        head_dim,
        group,
        page_size,
        max_top_k,
        always_tail,
        position_dtype,
        key_norm_eps,
        verify_rows: _, // a verify key write's rows, which its phase states
    } = *geometry;
    let read = match kind {
        Kind::KeyWrite(_) => format!(
            "{:?}",
            (
                head_dim,
                group,
                page_size,
                position_dtype,
                key_norm_eps.to_bits()
            )
        ),
        Kind::Select { .. } => format!(
            "{:?}",
            (heads, head_dim, group, page_size, max_top_k, always_tail)
        ),
        Kind::TopKMerge { .. } => format!("{:?}", (group, max_top_k, always_tail)),
        Kind::Compact => format!("{}", geometry.list_width()),
    };
    format!("{} {kind:?} {read}", entry.symbol())
}

/// What a head-sum entry launches: its symbol and kind, and the geometry
/// its kernel reads; the merge never reads the pages its traits name.
fn head_sum_launch(entry: &head_sum::Entry) -> String {
    let head_sum::Entry {
        profile: _,  // filters catalogs; no launch reads it
        name: _,     // in the symbol
        geometry,    // by kind, below
        kind,        // selects the op
        capacity: _, // bounds the rows the selection settles
    } = entry;
    let head_sum::Geometry {
        heads,
        head_dim,
        group,
        page_size,
        max_top_k,
        always_tail,
    } = *geometry;
    let read = match kind {
        head_sum::Kind::Select { .. } => format!(
            "{:?}",
            (heads, head_dim, group, page_size, max_top_k, always_tail)
        ),
        head_sum::Kind::TopKMerge { .. } => format!("{:?}", (group, max_top_k, always_tail)),
    };
    format!("{} {kind:?} {read}", entry.symbol())
}

/// What a block-id merge launches: its symbol, ranks and the pools and
/// top-k `topk_merge_blocks` reads; never the pages or tail its traits
/// also name (`cuda/pooled_indexer/block_merge.cuh`).
fn block_launch(entry: &block_merge::Entry) -> String {
    let block_merge::Entry {
        profile: _,  // filters catalogs; no launch reads it
        name: _,     // in the symbol
        geometry,    // below
        ranks,       // selects the op
        capacity: _, // bounds the rows the selection settles
    } = entry;
    let head_sum::Geometry {
        heads,
        head_dim,
        group,
        page_size: _, // in the traits text only
        max_top_k,
        always_tail: _, // in the traits text only
    } = *geometry;
    format!(
        "{} {ranks} {:?}",
        entry.symbol(),
        (heads, head_dim, group, max_top_k)
    )
}

/// A selection entry of either scoring.
#[derive(Clone, Copy, Debug)]
enum SelectEntry<'a> {
    Weighted(&'a Entry),
    HeadSum(&'a head_sum::Entry),
}

impl SelectEntry<'_> {
    fn describe(self) -> String {
        match self {
            Self::Weighted(entry) => format!("{entry:?}"),
            Self::HeadSum(entry) => format!("{entry:?}"),
        }
    }

    fn symbols(self) -> (String, Option<String>) {
        match self {
            Self::Weighted(entry) => (entry.symbol(), entry.merge_symbol()),
            Self::HeadSum(entry) => (entry.symbol(), entry.merge_symbol()),
        }
    }

    fn check_context_parallel(self, rank: u32, size: u32) -> Result<(), Refusal> {
        match self {
            Self::Weighted(entry) => entry.check_context_parallel(rank, size),
            Self::HeadSum(entry) => entry.check_context_parallel(rank, size),
        }
    }

    fn planned(self, rows: u32) -> Result<Planned, Refusal> {
        let (plan, merge, splits) = match self {
            Self::Weighted(entry) => (
                entry.plan(rows)?,
                entry.merge_plan(rows)?,
                entry.splits(rows),
            ),
            Self::HeadSum(entry) => (
                entry.plan(rows)?,
                entry.merge_plan(rows)?,
                entry.splits(rows),
            ),
        };
        Ok(Planned {
            plan,
            merge,
            splits,
        })
    }
}

/// A merge entry: a key list merge of either selection, or a block-id
/// merge.
#[derive(Clone, Copy, Debug)]
enum MergeEntry<'a> {
    Weighted(&'a Entry),
    HeadSum(&'a head_sum::Entry),
    Blocks(&'a block_merge::Entry),
}

impl MergeEntry<'_> {
    fn describe(self) -> String {
        match self {
            Self::Weighted(entry) => format!("{entry:?}"),
            Self::HeadSum(entry) => format!("{entry:?}"),
            Self::Blocks(entry) => format!("{entry:?}"),
        }
    }

    fn symbol(self) -> String {
        match self {
            Self::Weighted(entry) => entry.symbol(),
            Self::HeadSum(entry) => entry.symbol(),
            Self::Blocks(entry) => entry.symbol(),
        }
    }

    fn plan(self, rows: u32) -> Result<LaunchPlan, Refusal> {
        match self {
            Self::Weighted(entry) => entry.plan(rows),
            Self::HeadSum(entry) => entry.plan(rows),
            Self::Blocks(entry) => entry.plan(rows),
        }
    }
}

fn check_top_k(top_k: u32, group: u32, max_top_k: u32) -> Result<(), Refusal> {
    if top_k == 0 || top_k > max_top_k || !top_k.is_multiple_of(group) {
        return Err(Refusal(
            "runtime indexer top-k is outside the compiled maximum",
        ));
    }
    Ok(())
}

fn buffer(view: View) -> Argument {
    Argument::DeviceBuffer(view.handle)
}

/// `a * b * c` bytes, refused rather than wrapped.
fn bytes(a: u64, b: u64, c: u64) -> Result<u64, Refusal> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(c))
        .ok_or(Refusal("pooled indexer bytes overflow"))
}

/// `rows` rows of `width` elements of `element` bytes at a stride of
/// `stride` elements.
fn strided(rows: u64, width: u32, stride: u32, element: u64) -> Result<u64, Refusal> {
    if stride < width {
        return Err(Refusal("a row stride is narrower than the row"));
    }
    bytes(rows - 1, u64::from(stride), element)?
        .checked_add(u64::from(width) * element)
        .ok_or(Refusal("pooled indexer bytes overflow"))
}

/// Checks pooled key pages of `page_bytes` bytes each: the keys hold every
/// page at its stride, and the page tables every item's row.
fn check_pages(pages: &PooledKeyPages<View>, page_bytes: u64) -> Result<(), Refusal> {
    if pages.pages == 0 {
        return Err(Refusal("pooled keys need a page"));
    }
    if pages.page_stride < page_bytes {
        return Err(Refusal("a pooled key page stride is below the page's keys"));
    }
    let end = u64::from(pages.pages - 1)
        .checked_mul(pages.page_stride)
        .and_then(|start| start.checked_add(page_bytes))
        .ok_or(Refusal("pooled indexer bytes overflow"))?;
    pages.keys.covers(end)?;
    // The host bounds each row's item by the page tables' sequences.
    pages.page_tables.check()
}

/// One row count's launches: the main plan, and a selection's split merge
/// plan and splits.
#[derive(Clone, Copy)]
struct Planned {
    plan: LaunchPlan,
    merge: Option<LaunchPlan>,
    splits: u32,
}

/// An entry planned at every row count from one to the prepared capacity.
struct Plans {
    symbol: String,
    rows: Vec<LaunchPlan>,
    info: OpInfo,
}

impl Plans {
    /// Plans `symbol` with `plan` for every row count up to
    /// `need.capacity`, at most the entry's `capacity`. `describe` is the
    /// entry's debug form, which the identity digests.
    fn new(
        symbol: String,
        describe: &str,
        guarantees: Invariance,
        need: Need,
        plan: impl Fn(u32) -> Result<LaunchPlan, Refusal>,
    ) -> Result<Self, Refused> {
        let mut rows = Vec::with_capacity(need.capacity as usize);
        for count in 1..=need.capacity {
            let planned = plan(count)?;
            check_plan(&planned)?;
            rows.push(planned);
        }
        let first = rows
            .first()
            .ok_or(Refusal("a pooled indexer op serves rows"))?;
        let symbols = vec![(symbol.clone(), first.dynamic_shared_bytes)];
        let workspace = rows
            .iter()
            .map(|planned| planned.workspace_bytes)
            .max()
            .unwrap_or(0);
        let info = op_info(&symbol, describe, guarantees, symbols, workspace, need)?;
        Ok(Self { symbol, rows, info })
    }

    /// The rows of `counts` and their plans, refused outside the prepared
    /// capacity.
    fn at(&self, counts: Counts) -> Result<(u64, LaunchPlan), Refusal> {
        if counts.rows == 0 || counts.rows as usize > self.rows.len() {
            return Err(Refusal("pooled indexer rows exceed the prepared capacity"));
        }
        Ok((u64::from(counts.rows), self.rows[counts.rows as usize - 1]))
    }
}

/// The info of an op of entry `symbol` (whose debug form `describe` the
/// identity digests) that launches `symbols` and sizes `workspace` bytes,
/// checked against `need`.
fn op_info(
    symbol: &str,
    describe: &str,
    guarantees: Invariance,
    symbols: Vec<(String, u32)>,
    workspace: u64,
    need: Need,
) -> Result<OpInfo, Refused> {
    let digest = kernel_build::digest(describe.as_bytes());
    let mut info = OpInfo::new(
        format!("sm120.pooled_indexer.v1[{symbol}]#{}", &digest[..16]),
        guarantees,
        symbols,
    );
    if workspace > 0 {
        info.bytes.push(("workspace".into(), workspace));
    }
    need.check(&info)?;
    Ok(info)
}

/// The kernel that writes a selection's output at one row count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Writer {
    /// The entry's split merge, after its split kernel.
    Merge,
    /// The tree merge in the split merge's place.
    Tree,
    /// The radix merge in the split merge's place.
    Radix,
    /// The one-split form in place of the split kernel and its merge.
    OneSplit,
}

/// The writer of a selection launch of `splits` splits per row, given the
/// forms registered for its entry: the one-split form at one split, the
/// radix merge (else the tree merge) at two or more, else the entry's own
/// merge. Every writer writes the same bits.
pub(crate) fn writer(splits: u32, one_split: bool, tree: bool, radix: bool) -> Writer {
    match splits {
        1 if one_split => Writer::OneSplit,
        1 => Writer::Merge,
        _ if radix => Writer::Radix,
        _ if tree => Writer::Tree,
        _ => Writer::Merge,
    }
}

/// The form of a weighted selection `entry` among `forms` (each with the
/// selection it was made from, its symbol and what it launches) that
/// serves `capacity` rows, if the model registered one made from an entry
/// that launches as `entry` does. A form plans through the entry it was
/// made from, so it holds that entry's capacity: of the matching forms, the
/// least capacity that holds the rows wins, whatever the registration order
/// (`select::choose`); equal forms of that capacity are one, and two that
/// launch differently are a tie the op refuses. Matching forms that all
/// hold fewer rows refuse the op rather than leave it to the entry.
fn form<'a, F: std::fmt::Debug>(
    entry: &Entry,
    forms: &'a [F],
    made_from: impl Fn(&F) -> &Entry,
    describe: impl Fn(&F) -> (String, String),
    capacity: u32,
    family: &str,
) -> Result<Option<&'a F>, Refused> {
    let key = weighted_launch(entry);
    let found = forms
        .iter()
        .filter(|form| weighted_launch(made_from(form)) == key)
        .map(|form| {
            let (symbol, launches) = describe(form);
            Candidate::new(form, symbol, launches, Invariance::NONE)
                .capacity(made_from(form).capacity)
        })
        .collect::<Vec<_>>();
    if found.is_empty() {
        return Ok(None);
    }
    // Forms claim nothing of their own: the op's claims are its writers'.
    Ok(Some(choose(family, found, Need::rows(capacity))?.entry))
}

/// The forms registered for one weighted selection entry.
#[derive(Clone, Copy, Default)]
struct Forms<'a> {
    one_split: Option<&'a rows::Entry>,
    tree: Option<&'a merge_tree::Entry>,
    radix: Option<&'a merge_radix::Entry>,
}

impl<'a> Forms<'a> {
    /// The forms of `entry` among the selection's that serve `capacity`
    /// rows.
    fn of(selection: &'a Selection, entry: &Entry, capacity: u32) -> Result<Self, Refused> {
        Ok(Self {
            one_split: form(
                entry,
                &selection.one_split,
                |form| &form.select,
                |form| (form.symbol(), format!("{:?}", form.entry_spec())),
                capacity,
                "one-split pooled indexer selection",
            )?,
            tree: form(
                entry,
                &selection.trees,
                |form| &form.select,
                |form| (form.symbol(), format!("{:?}", form.entry_spec())),
                capacity,
                "pooled indexer tree merge",
            )?,
            radix: form(
                entry,
                &selection.radixes,
                |form| &form.select,
                |form| (form.symbol(), format!("{:?}", form.entry_spec())),
                capacity,
                "pooled indexer radix merge",
            )?,
        })
    }

    fn writer(&self, splits: u32) -> Writer {
        writer(
            splits,
            self.one_split.is_some(),
            self.tree.is_some(),
            self.radix.is_some(),
        )
    }
}

/// A selection's launches at one row count.
#[derive(Clone, Copy)]
struct SelectRow {
    writer: Writer,
    /// The split kernel's plan, or the one-split form's.
    plan: LaunchPlan,
    /// The merge's plan; none for the one-split form.
    merge: Option<LaunchPlan>,
    splits: u32,
}

/// A selection's kernels: the entry's split kernel and merge and its
/// registered forms, by symbol.
struct SelectKernels {
    select: String,
    merge: String,
    tree: Option<String>,
    radix: Option<String>,
    one_split: Option<String>,
}

impl SelectKernels {
    /// The symbols a row of `writer` launches, in launch order.
    fn launched(&self, writer: Writer) -> Vec<&str> {
        match writer {
            Writer::Merge => vec![&self.select, &self.merge],
            Writer::Tree => vec![&self.select, registered(&self.tree)],
            Writer::Radix => vec![&self.select, registered(&self.radix)],
            Writer::OneSplit => vec![registered(&self.one_split)],
        }
    }
}

/// The symbol of a form a row's writer runs, which the op registered.
fn registered(symbol: &Option<String>) -> &str {
    symbol
        .as_deref()
        .expect("a row's writer is a registered form")
}

/// The key write of `op` at context-parallel group rank `group_rank` of
/// `group_size` through the selection's key write entry for it.
pub(super) fn key_write(
    selection: &Selection,
    op: &PooledIndexerQuery,
    group_rank: u32,
    group_size: u32,
    need: Need,
) -> Result<Prepared<IndexerKeyWriteBuffers<View>>, Refused> {
    op.validate()?;
    let PooledIndexerQuery::KeyWrite {
        pooling,
        key_norm_eps_bits,
        phase,
        ..
    } = *op
    else {
        return Err(Refused::Rule(
            "an indexer key write takes a key write query".into(),
        ));
    };
    limits(op)?;
    let (entry, guarantees) = fitting(
        selection
            .weighted
            .iter()
            .filter(|entry| matches!(entry.kind, Kind::KeyWrite(_)) && serves(entry.query(), *op))
            .map(|entry| {
                let launches = weighted_launch(entry);
                (entry, entry.capacity, entry.symbol(), launches)
            }),
        // Its declaration (`pooled_indexer::Entry::invariance`), which
        // names its tail state's anchors.
        |entry| entry.invariance(),
        need,
    )?;
    entry.check_context_parallel(group_rank, group_size)?;
    let plans = Plans::new(
        entry.symbol(),
        &format!("{entry:?}"),
        guarantees,
        need,
        |rows| entry.plan(rows),
    )?;
    Ok(Box::new(KeyWrite {
        plans,
        width: pooling.width,
        group: pooling.group,
        position_bytes: if pooling.position_dtype == DType::F32 {
            4
        } else {
            2
        },
        state_bytes: pooling.tail_state_elements() * BF16,
        page_bytes: op
            .pooled_page_bytes()
            .expect("a key write has pooled pages"),
        verify_rows: match phase {
            RecurrentPhase::Verify { max_rows } => Some(max_rows),
            _ => None,
        },
        eps_bits: key_norm_eps_bits,
        cp: [group_rank, group_size],
    }))
}

/// The selection of `op` keeping `top_k` tokens at group rank `rank`,
/// through the selection's weighted or head-sum entry for it.
pub(super) fn select(
    selection: &Selection,
    claims: &Claims,
    op: &PooledIndexerQuery,
    top_k: u32,
    rank: u32,
    need: Need,
) -> Result<Prepared<IndexerSelectBuffers<View>>, Refused> {
    op.validate()?;
    let (weighted, heads, head_dim, group, max_top_k, mode, page_size) = match *op {
        PooledIndexerQuery::Select {
            heads,
            head_dim,
            group,
            max_top_k,
            mode,
            page_size,
            ..
        } => (true, heads, head_dim, group, max_top_k, mode, page_size),
        PooledIndexerQuery::HeadSumSelect {
            heads,
            head_dim,
            group,
            max_top_k,
            mode,
            page_size,
            ..
        } => (false, heads, head_dim, group, max_top_k, mode, page_size),
        _ => {
            return Err(Refused::Rule(
                "an indexer selection takes a selection query".into(),
            ));
        }
    };
    limits(op)?;
    check_top_k(top_k, group, max_top_k)?;
    let (cp_rank, cp_size) = match mode {
        IndexerSelectionMode::Global if rank != 0 => {
            return Err(Refused::Rule(
                "a global indexer selection runs at rank 0".into(),
            ));
        }
        IndexerSelectionMode::Global => (0, 1),
        IndexerSelectionMode::PerRank { ranks } => (rank, ranks),
    };
    // A selection launches its entry and its split merge, and its output
    // is the merge's pairs. The entry's per-split lists are workspace whose
    // bits follow the split count, which the prepared rows set, so the op
    // guarantees the claim the model declares for the composition: the
    // split merge's, the entry that writes the output.
    // A weighted entry declares the composition's claims itself
    // (`pooled_indexer::Entry::invariance`).
    let claimed = |entry: &SelectEntry| match entry {
        SelectEntry::Weighted(entry) => entry.invariance(),
        SelectEntry::HeadSum(_) => {
            let (symbol, merge) = entry.symbols();
            claims.of(&[merge.as_deref().unwrap_or(&symbol)])
        }
    };
    let (entry, guarantees) = if weighted {
        fitting(
            selection
                .weighted
                .iter()
                .filter(|entry| {
                    matches!(entry.kind, Kind::Select { .. }) && serves(entry.query(), *op)
                })
                .map(|entry| {
                    let launches = weighted_launch(entry);
                    (
                        SelectEntry::Weighted(entry),
                        entry.capacity,
                        entry.symbol(),
                        launches,
                    )
                }),
            claimed,
            need,
        )?
    } else {
        fitting(
            selection
                .head_sum
                .iter()
                .filter(|entry| {
                    matches!(entry.kind, head_sum::Kind::Select { .. })
                        && serves(entry.query(), *op)
                })
                .map(|entry| {
                    let launches = head_sum_launch(entry);
                    (
                        SelectEntry::HeadSum(entry),
                        entry.capacity,
                        entry.symbol(),
                        launches,
                    )
                }),
            claimed,
            need,
        )?
    };
    entry.check_context_parallel(cp_rank, cp_size)?;
    let forms = match entry {
        SelectEntry::Weighted(entry) => Forms::of(selection, entry, need.capacity)?,
        SelectEntry::HeadSum(_) => Forms::default(),
    };
    let (symbol, merge) = entry.symbols();
    let kernels = SelectKernels {
        select: symbol,
        merge: merge.expect("a selection entry has a split merge"),
        tree: forms.tree.map(merge_tree::Entry::symbol),
        radix: forms.radix.map(merge_radix::Entry::symbol),
        one_split: forms.one_split.map(rows::Entry::symbol),
    };
    let mut rows = Vec::with_capacity(need.capacity as usize);
    for count in 1..=need.capacity {
        let planned = entry.planned(count)?;
        let writer = forms.writer(planned.splits);
        let (plan, merge) = match (writer, forms) {
            (
                Writer::OneSplit,
                Forms {
                    one_split: Some(form),
                    ..
                },
            ) => (form.plan(count)?, None),
            (
                Writer::Tree,
                Forms {
                    tree: Some(form), ..
                },
            ) => (planned.plan, Some(form.plan(count)?)),
            (
                Writer::Radix,
                Forms {
                    radix: Some(form), ..
                },
            ) => (planned.plan, Some(form.plan(count)?)),
            _ => (planned.plan, planned.merge),
        };
        check_plan(&plan)?;
        if let Some(merge) = merge {
            check_plan(&merge)?;
        }
        rows.push(SelectRow {
            writer,
            plan,
            merge,
            splits: planned.splits,
        });
    }
    if rows.is_empty() {
        return Err(Refusal("a pooled indexer op serves rows").into());
    }
    // Each symbol in the order the launches first run it over ascending
    // row counts, with the most dynamic shared memory its plans take.
    let mut symbols: Vec<(String, u32)> = Vec::new();
    for row in &rows {
        let plans = [Some(row.plan), row.merge];
        for (symbol, plan) in kernels
            .launched(row.writer)
            .into_iter()
            .zip(plans.into_iter().flatten())
        {
            match symbols.iter_mut().find(|(held, _)| held == symbol) {
                Some((_, bytes)) => *bytes = (*bytes).max(plan.dynamic_shared_bytes),
                None => symbols.push((symbol.to_owned(), plan.dynamic_shared_bytes)),
            }
        }
    }
    // The op guarantees what every kernel that writes its output claims: a
    // weighted entry's forms their declarations, which all name the one-row
    // launch's bits, so their meet is the routed claim over the row counts.
    let mut writers = rows
        .iter()
        .map(|row| *kernels.launched(row.writer).last().expect("a row launches"))
        .filter(|writer| *writer != kernels.merge)
        .collect::<Vec<_>>();
    writers.dedup();
    let declared = [
        forms
            .one_split
            .map(|form| (form.symbol(), form.invariance())),
        forms.tree.map(|form| (form.symbol(), form.invariance())),
        forms.radix.map(|form| (form.symbol(), form.invariance())),
    ];
    let guarantees = if writers.is_empty() {
        guarantees
    } else if let SelectEntry::Weighted(_) = entry {
        declared
            .iter()
            .flatten()
            .filter(|(symbol, _)| writers.contains(&symbol.as_str()))
            .fold(guarantees, |held, &(_, claim)| held.meet(claim))
    } else {
        guarantees.meet(claims.of(&writers))
    };
    let workspace = rows
        .iter()
        .map(|row| row.plan.workspace_bytes)
        .max()
        .unwrap_or(0);
    let info = op_info(
        &kernels.select,
        &entry.describe(),
        guarantees,
        symbols,
        workspace,
        need,
    )?;
    let output_row = match mode {
        IndexerSelectionMode::Global => {
            u64::from(op.list_width().expect("a valid selection's list width")) * I32
        }
        IndexerSelectionMode::PerRank { .. } => u64::from(max_top_k / group) * PAIR,
    };
    Ok(Box::new(Select {
        rows,
        kernels,
        info,
        weighted,
        heads,
        head_dim,
        page_bytes: u64::from(page_size / group) * u64::from(head_dim) * BF16,
        output_row,
        top_k,
        cp: [cp_rank, cp_size],
    }))
}

/// The merge of `op` keeping `top_k` tokens through the selection's key
/// list merge or block-id merge entry for it.
pub(super) fn merge(
    selection: &Selection,
    claims: &Claims,
    op: &ContextParallelQuery,
    top_k: u32,
    need: Need,
) -> Result<Prepared<IndexerMergeBuffers<View>>, Refused> {
    op.validate()?;
    let (ranks, group, max_top_k, blocks) = match *op {
        ContextParallelQuery::TopKMerge {
            ranks,
            group,
            max_top_k,
            ..
        } => (ranks, group, max_top_k, false),
        ContextParallelQuery::TopKMergeBlocks {
            ranks,
            group,
            max_top_k,
            ..
        } => (ranks, group, max_top_k, true),
        _ => {
            return Err(Refused::Rule(
                "an indexer merge takes a top-k merge query".into(),
            ));
        }
    };
    check_top_k(top_k, group, max_top_k)?;
    // A weighted and a head-sum key list merge of equal shape are two
    // entries for one op: `fitting` refuses the pair rather than pick one.
    let weighted = selection
        .weighted
        .iter()
        .filter(|entry| matches!(entry.kind, Kind::TopKMerge { .. }))
        .filter(|entry| serves_merge(entry.query(), *op))
        .map(|entry| {
            let launches = weighted_launch(entry);
            (
                MergeEntry::Weighted(entry),
                entry.capacity,
                entry.symbol(),
                launches,
            )
        });
    let head_sum = selection
        .head_sum
        .iter()
        .filter(|entry| matches!(entry.kind, head_sum::Kind::TopKMerge { .. }))
        .filter(|entry| serves_merge(entry.query(), *op))
        .map(|entry| {
            let launches = head_sum_launch(entry);
            (
                MergeEntry::HeadSum(entry),
                entry.capacity,
                entry.symbol(),
                launches,
            )
        });
    let block_merges = selection
        .block_merges
        .iter()
        .filter(|entry| uncapped_merge(entry.query()) == uncapped_merge(*op))
        .map(|entry| {
            let launches = block_launch(entry);
            (
                MergeEntry::Blocks(entry),
                entry.capacity,
                entry.symbol(),
                launches,
            )
        });
    let (entry, guarantees) = fitting(
        weighted.chain(head_sum).chain(block_merges),
        |entry| match entry {
            MergeEntry::Weighted(entry) => entry.invariance(),
            _ => claims.of(&[&entry.symbol()]),
        },
        need,
    )?;
    let plans = Plans::new(
        entry.symbol(),
        &entry.describe(),
        guarantees,
        need,
        |rows| entry.plan(rows),
    )?;
    let output_row = if blocks {
        u64::from(max_top_k / group) * I32
    } else {
        u64::from(op.list_width().expect("a valid merge's list width")) * I32
    };
    Ok(Box::new(Merge {
        plans,
        ranks,
        kept: max_top_k / group,
        blocks,
        output_row,
        top_k,
    }))
}

/// The compaction of `op` through the selection's compaction entry for it.
pub(super) fn compact(
    selection: &Selection,
    op: &PooledIndexerQuery,
    need: Need,
) -> Result<Prepared<SelectionCompactBuffers<View>>, Refused> {
    op.validate()?;
    let PooledIndexerQuery::CompactSelection { width, .. } = *op else {
        return Err(Refused::Rule(
            "a selection compaction takes a compaction query".into(),
        ));
    };
    let (entry, guarantees) = fitting(
        selection
            .weighted
            .iter()
            .filter(|entry| entry.kind == Kind::Compact && serves(entry.query(), *op))
            .map(|entry| {
                let launches = weighted_launch(entry);
                (entry, entry.capacity, entry.symbol(), launches)
            }),
        |entry| entry.invariance(),
        need,
    )?;
    let plans = Plans::new(
        entry.symbol(),
        &format!("{entry:?}"),
        guarantees,
        need,
        |rows| entry.plan(rows),
    )?;
    Ok(Box::new(Compact {
        plans,
        row: u64::from(width) * I32,
    }))
}

/// `key_write`: a warp per row; the first row of each item pools the group
/// straddling the launch start and writes the item's tail state.
struct KeyWrite {
    plans: Plans,
    width: u32,
    group: u32,
    position_bytes: u64,
    /// Bytes of one tail state entry.
    state_bytes: u64,
    /// Bytes of one page's pooled keys.
    page_bytes: u64,
    /// The rows per item of a verify write, which keeps checkpoints.
    verify_rows: Option<u32>,
    eps_bits: u32,
    /// The launch's context-parallel group rank and size.
    cp: [u32; 2],
}

impl PreparedOp<IndexerKeyWriteBuffers<View>> for KeyWrite {
    fn info(&self) -> &OpInfo {
        &self.plans.info
    }

    fn launches(
        &self,
        buffers: &IndexerKeyWriteBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (rows, plan) = self.plans.at(counts)?;
        let width = u64::from(self.width);
        buffers
            .key
            .covers(strided(rows, self.width, buffers.key_stride, BF16)?)?;
        buffers
            .gate
            .covers(strided(rows, self.width, buffers.gate_stride, BF16)?)?;
        buffers.norm_weight.covers(width * BF16)?;
        buffers.norm_bias.covers(width * BF16)?;
        buffers
            .position
            .covers(u64::from(self.group) * width * self.position_bytes)?;
        buffers.row_batch.covers(rows * I32)?;
        buffers.positions.covers(rows * I32)?;
        buffers.live_rows.covers(4)?;
        let tail = &buffers.tail;
        if tail.entries == 0 {
            return Err(Refusal("an indexer tail state holds at least one entry"));
        }
        tail.state
            .covers(bytes(u64::from(tail.entries), self.state_bytes, 1)?)?;
        // The host bounds each row's item by the per-item arrays' rows.
        tail.slot_ids.covers(I32)?;
        let checkpoints = match (self.verify_rows, tail.checkpoint_ids) {
            (Some(rows), Some(ids)) => {
                ids.covers(u64::from(rows) * I32)?;
                Some(buffer(ids))
            }
            (Some(_), None) => {
                return Err(Refusal("a verify key write needs checkpoint ids"));
            }
            // A prefill or decode write keeps no checkpoints and ignores
            // the ids, as the builder's parameters leave them out.
            (None, _) => None,
        };
        check_pages(&buffers.pooled, self.page_bytes)?;
        let mut arguments = vec![
            buffer(buffers.key),
            Argument::U32(buffers.key_stride),
            buffer(buffers.gate),
            Argument::U32(buffers.gate_stride),
            buffer(buffers.norm_weight),
            buffer(buffers.norm_bias),
            buffer(buffers.position),
            buffer(buffers.row_batch),
            buffer(buffers.positions),
            buffer(buffers.live_rows),
            buffer(tail.slot_ids),
        ];
        arguments.extend(checkpoints);
        arguments.extend([
            buffer(tail.state),
            buffer(buffers.pooled.page_tables.tables),
            buffer(buffers.pooled.keys),
            Argument::U32(buffers.pooled.page_tables.max_pages),
            Argument::U64(buffers.pooled.page_stride),
            Argument::U32(self.eps_bits),
            Argument::U32(self.cp[0]),
            Argument::U32(self.cp[1]),
        ]);
        Ok(vec![kernel_launch(
            &self.plans.symbol,
            rows,
            plan,
            arguments,
        )])
    }
}

/// `select_split` or `head_sum_split`, grid `[rows, splits]`, then its
/// split merge (or the tree or radix merge in its place), grid `[rows]`;
/// or the one-split form alone, a CTA per row.
struct Select {
    /// The launches at each row count from one.
    rows: Vec<SelectRow>,
    kernels: SelectKernels,
    info: OpInfo,
    /// A weighted selection, which reads head weights.
    weighted: bool,
    heads: u32,
    head_dim: u32,
    page_bytes: u64,
    /// Bytes of one output row: a key list or a rank's pairs.
    output_row: u64,
    top_k: u32,
    /// The launch's context-parallel group rank and size.
    cp: [u32; 2],
}

/// A launch of `symbol` over `rows` rows.
fn kernel_launch(symbol: &str, rows: u64, plan: LaunchPlan, arguments: Vec<Argument>) -> Launch {
    Launch {
        symbol: symbol.to_owned(),
        rows: rows as u32,
        plan,
        arguments: ArgumentList(arguments),
        collective: false,
        persistent: false,
    }
}

impl PreparedOp<IndexerSelectBuffers<View>> for Select {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &IndexerSelectBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows == 0 || counts.rows as usize > self.rows.len() {
            return Err(Refusal("pooled indexer rows exceed the prepared capacity"));
        }
        let (rows, row) = (u64::from(counts.rows), self.rows[counts.rows as usize - 1]);
        let heads = u64::from(self.heads);
        buffers
            .q
            .covers(bytes(rows, heads * u64::from(self.head_dim), BF16)?)?;
        let weights = match (self.weighted, buffers.head_weights) {
            (true, Some(weights)) => {
                weights.covers(bytes(rows, heads, BF16)?)?;
                Some(buffer(weights))
            }
            (true, None) => {
                return Err(Refusal("a weighted indexer selection needs head weights"));
            }
            // A head-sum selection reads no head weights and ignores them.
            (false, _) => None,
        };
        check_pages(&buffers.pooled, self.page_bytes)?;
        buffers.row_batch.covers(rows * I32)?;
        buffers.positions.covers(rows * I32)?;
        buffers.live_rows.covers(4)?;
        buffers.workspace.covers(row.plan.workspace_bytes)?;
        buffers.output.covers(bytes(rows, self.output_row, 1)?)?;
        let launched = self.kernels.launched(row.writer);
        let mut arguments = vec![buffer(buffers.q)];
        arguments.extend(weights);
        arguments.extend([
            buffer(buffers.pooled.keys),
            buffer(buffers.pooled.page_tables.tables),
            buffer(buffers.row_batch),
            buffer(buffers.positions),
            buffer(buffers.live_rows),
        ]);
        if row.writer == Writer::OneSplit {
            // The one-split form writes the output in the workspace's
            // place and takes no split count.
            arguments.extend([
                buffer(buffers.output),
                Argument::U32(buffers.pooled.page_tables.max_pages),
                Argument::U64(buffers.pooled.page_stride),
                Argument::U32(self.top_k),
                Argument::U32(self.cp[0]),
                Argument::U32(self.cp[1]),
            ]);
            return Ok(vec![kernel_launch(launched[0], rows, row.plan, arguments)]);
        }
        arguments.extend([
            buffer(buffers.workspace),
            Argument::U32(buffers.pooled.page_tables.max_pages),
            Argument::U64(buffers.pooled.page_stride),
            Argument::U32(self.top_k),
            Argument::U32(row.splits),
            Argument::U32(self.cp[0]),
            Argument::U32(self.cp[1]),
        ]);
        let merge = row.merge.expect("a split selection row has a merge plan");
        Ok(vec![
            kernel_launch(launched[0], rows, row.plan, arguments),
            // The split merge, or the tree or radix merge of its ABI.
            kernel_launch(
                launched[1],
                rows,
                merge,
                vec![
                    buffer(buffers.workspace),
                    buffer(buffers.positions),
                    buffer(buffers.live_rows),
                    buffer(buffers.output),
                    Argument::U32(self.top_k),
                    Argument::U32(row.splits),
                ],
            ),
        ])
    }
}

/// `topk_merge` or `topk_merge_blocks`: a CTA per row over the gathered
/// pairs of every rank.
struct Merge {
    plans: Plans,
    ranks: u32,
    /// Pairs of one rank's row.
    kept: u32,
    /// A block-id merge, which reads no positions.
    blocks: bool,
    output_row: u64,
    top_k: u32,
}

impl PreparedOp<IndexerMergeBuffers<View>> for Merge {
    fn info(&self) -> &OpInfo {
        &self.plans.info
    }

    fn launches(
        &self,
        buffers: &IndexerMergeBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (rows, plan) = self.plans.at(counts)?;
        if u64::from(buffers.rank_rows) < rows {
            return Err(Refusal("each rank's gathered pairs hold the launch's rows"));
        }
        let pairs = bytes(u64::from(self.ranks), u64::from(buffers.rank_rows), 1)?;
        buffers
            .pairs
            .covers(bytes(pairs, u64::from(self.kept), PAIR)?)?;
        buffers.live_rows.covers(4)?;
        buffers.output.covers(bytes(rows, self.output_row, 1)?)?;
        let mut arguments = vec![buffer(buffers.pairs)];
        match (self.blocks, buffers.positions) {
            (false, Some(positions)) => {
                positions.covers(rows * I32)?;
                arguments.push(buffer(positions));
            }
            (false, None) => {
                return Err(Refusal("a key list merge reads each row's position"));
            }
            // A block-id merge reads no positions and ignores them.
            (true, _) => {}
        }
        arguments.extend([
            buffer(buffers.live_rows),
            buffer(buffers.output),
            Argument::U32(self.top_k),
            Argument::U32(buffers.rank_rows),
        ]);
        Ok(vec![kernel_launch(
            &self.plans.symbol,
            rows,
            plan,
            arguments,
        )])
    }
}

/// `compact_selection`: a CTA per output row.
struct Compact {
    plans: Plans,
    /// Bytes of one key list row.
    row: u64,
}

impl PreparedOp<SelectionCompactBuffers<View>> for Compact {
    fn info(&self) -> &OpInfo {
        &self.plans.info
    }

    fn launches(
        &self,
        buffers: &SelectionCompactBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (rows, plan) = self.plans.at(counts)?;
        // The host bounds each source row by the lists' rows.
        buffers.lists.covers(self.row)?;
        buffers.source.covers(rows * I32)?;
        buffers.live_rows.covers(4)?;
        buffers.output.covers(bytes(rows, self.row, 1)?)?;
        Ok(vec![kernel_launch(
            &self.plans.symbol,
            rows,
            plan,
            vec![
                buffer(buffers.lists),
                buffer(buffers.source),
                buffer(buffers.live_rows),
                buffer(buffers.output),
            ],
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use crate::pooled_indexer::{Geometry, entries, merge_radix, merge_tree, rows};
    use crate::{Profile, ProfileNames};
    use kernel_build::EntryParam;
    use kernel_core::attention::BlockTables;
    use kernel_core::backend::Backend;
    use kernel_core::gated_pooling::GatedGroupPooling;
    use kernel_core::pooled_indexer::{IndexerTailState, PooledKeyPages};

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "pooled-indexer-backend-test",
        tag: "pooled_indexer_backend_test",
    });

    /// The weighted indexer's shapes: 32 heads of 128, pools of 4 with the
    /// always-selected tail, top 2048 tokens, 64-token pages and verify
    /// items of 6 rows.
    fn geometry() -> Geometry {
        Geometry {
            heads: 32,
            head_dim: 128,
            group: 4,
            page_size: 64,
            max_top_k: 2048,
            always_tail: true,
            position_dtype: DType::Bf16,
            key_norm_eps: 1e-6,
            verify_rows: 6,
        }
    }

    /// The head-sum selection's shapes: four heads of 128 over the same
    /// pools and pages.
    fn head_sum_geometry() -> head_sum::Geometry {
        head_sum::Geometry {
            heads: 4,
            head_dim: 128,
            group: 4,
            page_size: 64,
            max_top_k: 2048,
            always_tail: true,
        }
    }

    fn backend() -> Sm120 {
        Sm120::new()
            .with_pooled_indexer(entries(PROFILE, geometry()))
            .with_head_sum_indexer(head_sum::entries(PROFILE, head_sum_geometry()))
            .with_block_merges(block_merge::entries(PROFILE, head_sum_geometry(), &[2, 4]))
    }

    fn entry(kind: Kind) -> Entry {
        entries(PROFILE, geometry())
            .into_iter()
            .find(|entry| entry.kind == kind)
            .unwrap()
    }

    fn head_sum_entry(kind: head_sum::Kind) -> head_sum::Entry {
        head_sum::entries(PROFILE, head_sum_geometry())
            .into_iter()
            .find(|entry| entry.kind == kind)
            .unwrap()
    }

    /// Arguments in `params` order, each bound by name as the model's
    /// lowering binds them.
    fn by_name(params: &[EntryParam], named: &[(&str, Argument)]) -> ArgumentList {
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
                    named
                        .iter()
                        .find(|(key, _)| key == name)
                        .unwrap_or_else(|| panic!("no binding for {name}"))
                        .1
                        .clone()
                })
                .collect(),
        )
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

    const ROWS: u64 = 2048;
    const PAGES: u32 = 100;
    const MAX_PAGES: u32 = 1024;
    /// Pooled keys follow each page's latent rows in the KV page.
    const PAGE_STRIDE: u64 = 64 * 1152 + 16 * 128 * 2;

    fn pages() -> PooledKeyPages<View> {
        PooledKeyPages {
            keys: view(20, u64::from(PAGES) * PAGE_STRIDE),
            pages: PAGES,
            page_stride: PAGE_STRIDE,
            page_tables: BlockTables {
                tables: view(21, 32 * u64::from(MAX_PAGES) * 4),
                sequences: 32,
                max_pages: MAX_PAGES,
            },
        }
    }

    fn page_arguments() -> [(&'static str, Argument); 4] {
        [
            ("pooled_keys", Argument::DeviceBuffer(20)),
            ("page_tables", Argument::DeviceBuffer(21)),
            ("max_pages", Argument::U32(MAX_PAGES)),
            ("page_stride_bytes", Argument::U64(PAGE_STRIDE)),
        ]
    }

    fn key_write_op(phase: RecurrentPhase, capacity: u32) -> PooledIndexerQuery {
        let StaticQuery::PooledIndexer(query) = entry(Kind::KeyWrite(phase)).query() else {
            unreachable!()
        };
        let PooledIndexerQuery::KeyWrite {
            pooling,
            key_norm_eps_bits,
            page_size,
            key_format,
            phase,
            ..
        } = query
        else {
            unreachable!()
        };
        PooledIndexerQuery::KeyWrite {
            pooling,
            key_norm_eps_bits,
            page_size,
            key_format,
            phase,
            capacity,
        }
    }

    fn key_write_buffers(checkpoints: bool) -> IndexerKeyWriteBuffers<View> {
        IndexerKeyWriteBuffers {
            key: view(1, ROWS * 128 * 2),
            key_stride: 128,
            gate: view(2, ROWS * 128 * 2),
            gate_stride: 128,
            norm_weight: view(3, 256),
            norm_bias: view(4, 256),
            position: view(5, 4 * 256),
            row_batch: view(6, ROWS * 4),
            positions: view(7, ROWS * 4),
            live_rows: view(8, 4),
            tail: IndexerTailState {
                state: view(9, 300 * 768 * 2),
                entries: 300,
                slot_ids: view(10, 32 * 4),
                checkpoint_ids: checkpoints.then(|| view(11, 32 * 6 * 4)),
            },
            pooled: pages(),
        }
    }

    /// Every phase's key write at group rank 0 of 1 and at ranks of
    /// groups of 2 and 4 launches the entry as the builder binds it, with
    /// the group rank and size as launch scalars.
    #[test]
    fn key_writes_launch_what_the_builder_plans() {
        let backend = backend();
        let verify = RecurrentPhase::Verify { max_rows: 6 };
        for (phase, capacity, counts) in [
            (RecurrentPhase::Prefill, 2048, &[1, 7, 512, 2048][..]),
            (RecurrentPhase::Decode, 32, &[1, 5, 32][..]),
            (verify, 192, &[6, 30, 192][..]),
        ] {
            let entry = entry(Kind::KeyWrite(phase));
            for [rank, size] in [[0, 1], [1, 2], [0, 4], [3, 4]] {
                let prepared = backend
                    .indexer_key_write(
                        &key_write_op(phase, capacity),
                        rank,
                        size,
                        Need::rows(capacity),
                    )
                    .unwrap();
                let info = prepared.info();
                // It claims its declaration: a prefill key write any chunk
                // split, row independence and empty state at anchors on
                // its group; a verify key write one-row bits; a decode key
                // write nothing. None keeps a spare entry.
                assert_eq!(info.guarantees, entry.invariance());
                let aligned = |granule| {
                    Need::rows(capacity).with_exact(Invariance {
                        anchors: kernel_core::properties::Anchors::Aligned { granule },
                        ..Invariance::NONE
                    })
                };
                let op = key_write_op(phase, capacity);
                assert_eq!(
                    backend
                        .indexer_key_write(&op, rank, size, aligned(64))
                        .is_ok(),
                    phase == RecurrentPhase::Prefill
                );
                for granule in [0, 2, 6] {
                    assert!(matches!(
                        backend.indexer_key_write(&op, rank, size, aligned(granule)),
                        Err(Refused::Inexact { .. })
                    ));
                }
                for anchors in [
                    kernel_core::properties::Anchors::Spare,
                    kernel_core::properties::Anchors::SpareAfter { rows: 64 },
                ] {
                    let need = Need::rows(capacity).with_exact(Invariance {
                        anchors,
                        ..Invariance::NONE
                    });
                    assert!(matches!(
                        backend.indexer_key_write(&key_write_op(phase, capacity), rank, size, need),
                        Err(Refused::Inexact { .. })
                    ));
                }
                assert!(
                    info.implementation
                        .starts_with(&format!("sm120.pooled_indexer.v1[{}]#", entry.symbol()))
                );
                assert_eq!(info.symbols, [(entry.symbol(), 0)]);
                assert_eq!(info.bytes, []);
                assert!(info.weight_layouts.is_empty());
                let buffers = key_write_buffers(phase == verify);
                for &rows in counts {
                    let launches = prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap();
                    let mut named = vec![
                        ("key", Argument::DeviceBuffer(1)),
                        ("key_stride", Argument::U32(128)),
                        ("gate", Argument::DeviceBuffer(2)),
                        ("gate_stride", Argument::U32(128)),
                        ("norm_weight", Argument::DeviceBuffer(3)),
                        ("norm_bias", Argument::DeviceBuffer(4)),
                        ("position", Argument::DeviceBuffer(5)),
                        ("row_batch", Argument::DeviceBuffer(6)),
                        ("positions", Argument::DeviceBuffer(7)),
                        ("live_rows", Argument::DeviceBuffer(8)),
                        ("tail_state", Argument::DeviceBuffer(9)),
                        ("slot_ids", Argument::DeviceBuffer(10)),
                        ("checkpoint_ids", Argument::DeviceBuffer(11)),
                        ("eps_bits", Argument::U32(1e-6_f32.to_bits())),
                        ("cp_rank", Argument::U32(rank)),
                        ("cp_size", Argument::U32(size)),
                    ];
                    named.extend(page_arguments());
                    let legacy = launch(
                        entry.symbol(),
                        rows,
                        entry.plan(rows).unwrap(),
                        by_name(&entry.params(), &named),
                    );
                    assert_eq!(
                        launches,
                        [legacy],
                        "{phase:?} at {rows} rows, rank {rank} of {size}"
                    );
                }
            }
        }
    }

    /// The facts a drafter's verify path relies on: its key write is the
    /// prefill op prepared for the verify step's rows, which launches the
    /// prefill entry (no checkpoints) at each of them, and its share
    /// compaction is one op of 32 rows launched at 32 whatever the phase.
    #[test]
    fn a_verify_step_runs_the_prefill_key_write_and_a_32_row_compaction() {
        let backend = backend();
        let verify_rows = 32 * 6;
        let prefill = entry(Kind::KeyWrite(RecurrentPhase::Prefill));
        let write = backend
            .indexer_key_write(
                &key_write_op(RecurrentPhase::Prefill, 2048),
                0,
                1,
                Need::rows(verify_rows),
            )
            .unwrap();
        assert_eq!(write.info().symbols, [(prefill.symbol(), 0)]);
        let buffers = key_write_buffers(true);
        for rows in [1, 6, 30, verify_rows] {
            let launches = write
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].plan, prefill.plan(rows).unwrap());
            // No checkpoint ids: the prefill entry's ABI.
            assert_eq!(launches[0].arguments.0.len(), prefill.params().len());
            assert!(
                !launches[0]
                    .arguments
                    .0
                    .contains(&Argument::DeviceBuffer(11))
            );
        }
        let op = PooledIndexerQuery::CompactSelection {
            width: 2051,
            capacity: 32,
        };
        let compact = backend.indexer_compact(&op, Need::rows(32)).unwrap();
        let buffers = SelectionCompactBuffers {
            lists: view(1, 2048 * 2051 * 4),
            source: view(2, 32 * 4),
            live_rows: view(3, 4),
            output: view(4, 32 * 2051 * 4),
        };
        let launches = compact
            .launches(&buffers, View::EMPTY, Counts::rows(32))
            .unwrap();
        assert_eq!(launches[0].plan, entry(Kind::Compact).plan(32).unwrap());
    }

    /// A key write takes a group of 1 to 8 ranks and a rank inside it, as
    /// its entry does.
    #[test]
    fn key_writes_refuse_ranks_outside_their_group() {
        let backend = backend();
        let op = key_write_op(RecurrentPhase::Decode, 32);
        for [rank, size] in [[1, 1], [2, 2], [0, 0], [0, 9]] {
            assert!(
                matches!(
                    backend.indexer_key_write(&op, rank, size, Need::rows(32)),
                    Err(Refused::Rule(rule)) if rule.contains("rank or size")
                ),
                "rank {rank} of {size}"
            );
        }
        for size in 1..=8 {
            backend
                .indexer_key_write(&op, size - 1, size, Need::rows(32))
                .unwrap();
        }
    }

    fn select_buffers(weights: bool, output: u64) -> IndexerSelectBuffers<View> {
        IndexerSelectBuffers {
            q: view(1, ROWS * 32 * 128 * 2),
            head_weights: weights.then(|| view(2, ROWS * 32 * 2)),
            pooled: pages(),
            row_batch: view(6, ROWS * 4),
            positions: view(7, ROWS * 4),
            live_rows: view(8, 4),
            workspace: view(9, 8 << 20),
            output: view(10, ROWS * output),
        }
    }

    /// The selection and its split merge as the builder plans them.
    fn legacy_select(
        symbols: (String, Option<String>),
        params: [Vec<EntryParam>; 2],
        planned: Planned,
        rows: u32,
        weights: bool,
        cp: [u32; 2],
    ) -> Vec<Launch> {
        let mut named = vec![
            ("q", Argument::DeviceBuffer(1)),
            ("row_batch", Argument::DeviceBuffer(6)),
            ("positions", Argument::DeviceBuffer(7)),
            ("live_rows", Argument::DeviceBuffer(8)),
            ("workspace", Argument::DeviceBuffer(9)),
            ("output", Argument::DeviceBuffer(10)),
            ("top_k", Argument::U32(1024)),
            ("splits", Argument::U32(planned.splits)),
            ("cp_rank", Argument::U32(cp[0])),
            ("cp_size", Argument::U32(cp[1])),
        ];
        if weights {
            named.push(("head_weights", Argument::DeviceBuffer(2)));
        }
        named.extend(page_arguments());
        vec![
            launch(symbols.0, rows, planned.plan, by_name(&params[0], &named)),
            launch(
                symbols.1.unwrap(),
                rows,
                planned.merge.unwrap(),
                by_name(&params[1], &named),
            ),
        ]
    }

    fn select_op(mode: IndexerSelectionMode) -> PooledIndexerQuery {
        PooledIndexerQuery::Select {
            heads: 32,
            head_dim: 128,
            group: 4,
            max_top_k: 2048,
            always_tail: true,
            granularity: kernel_core::pooled_indexer::IndexerGranularity::Pool,
            key_format: kernel_core::pooled_indexer::IndexerKeyFormat::Bf16,
            mode,
            page_size: 64,
            capacity: 2048,
        }
    }

    fn head_sum_op(mode: IndexerSelectionMode) -> PooledIndexerQuery {
        PooledIndexerQuery::HeadSumSelect {
            heads: 4,
            head_dim: 128,
            group: 4,
            max_top_k: 2048,
            always_tail: true,
            mode,
            page_size: 64,
            capacity: 2048,
        }
    }

    const SELECT_ROWS: [u32; 6] = [1, 6, 32, 192, 1000, 2048];

    /// A head-sum selection guarantees the claim the model declares for
    /// its split merge, which writes the composition's output, whatever
    /// the entry declares; a block-id merge its entry's; anything
    /// undeclared claims nothing.
    #[test]
    fn selections_and_merges_report_declared_claims() {
        let claim = Invariance {
            chunk: kernel_core::properties::Chunk::Any,
            rows: kernel_core::properties::Rows::OneRow { max: 4 },
            ..Invariance::NONE
        };
        let select = head_sum_entry(head_sum::Kind::Select { ranks: Some(2) });
        let blocks = block_merge::entries(PROFILE, head_sum_geometry(), &[2])
            .pop()
            .unwrap();
        let merge_op = ContextParallelQuery::TopKMergeBlocks {
            ranks: 2,
            group: 4,
            max_top_k: 2048,
            capacity: 2048,
        };
        let op = head_sum_op(IndexerSelectionMode::PerRank { ranks: 2 });
        let wider = Invariance::one_row(8);
        let claimed = |symbols: Vec<String>| {
            backend().with_context_parallel_claims(symbols.into_iter().map(|s| (s, claim)))
        };
        // Both symbols of the selection declared.
        let both = claimed(vec![
            select.symbol(),
            select.merge_symbol().unwrap(),
            blocks.symbol(),
        ]);
        let prepared = both
            .indexer_select(&op, 1024, 1, Need::rows(4).with_exact(claim))
            .unwrap();
        assert_eq!(prepared.info().guarantees, claim);
        assert!(matches!(
            both.indexer_select(&op, 1024, 1, Need::rows(8).with_exact(wider)),
            Err(Refused::Inexact { .. })
        ));
        let merged = both
            .indexer_merge(&merge_op, 2048, Need::rows(4).with_exact(claim))
            .unwrap();
        assert_eq!(merged.info().guarantees, claim);
        // The entry undeclared (its per-split lists follow the split
        // count): the selection claims the composition's, its merge's.
        let composed = claimed(vec![select.merge_symbol().unwrap()]);
        let prepared = composed
            .indexer_select(&op, 1024, 1, Need::rows(4).with_exact(claim))
            .unwrap();
        assert_eq!(prepared.info().guarantees, claim);
        // The split merge undeclared: the selection claims nothing.
        let half = claimed(vec![select.symbol()]);
        let prepared = half.indexer_select(&op, 1024, 1, Need::rows(4)).unwrap();
        assert_eq!(prepared.info().guarantees, Invariance::NONE);
        assert!(matches!(
            half.indexer_select(&op, 1024, 1, Need::rows(4).with_exact(claim)),
            Err(Refused::Inexact { .. })
        ));
        assert_eq!(
            backend()
                .indexer_merge(&merge_op, 2048, Need::rows(4))
                .unwrap()
                .info()
                .guarantees,
            Invariance::NONE
        );
    }

    #[test]
    fn selections_launch_what_the_builder_plans() {
        let backend = backend();
        for (mode, rank, ranks, output) in [
            (IndexerSelectionMode::Global, 0, None, 2051 * 4),
            (
                IndexerSelectionMode::PerRank { ranks: 2 },
                1,
                Some(2),
                512 * 8,
            ),
            (
                IndexerSelectionMode::PerRank { ranks: 4 },
                3,
                Some(4),
                512 * 8,
            ),
        ] {
            let cp = [rank, ranks.unwrap_or(1)];
            // The weighted selection.
            let prepared = backend
                .indexer_select(&select_op(mode), 1024, rank, Need::rows(2048))
                .unwrap();
            let entry = entry(Kind::Select { ranks });
            let info = prepared.info();
            assert_eq!(
                info.symbols,
                [(entry.symbol(), 58_256), (entry.merge_symbol().unwrap(), 0)]
            );
            let workspace = (1..=2048).map(|rows| entry.workspace_bytes(rows)).max();
            assert_eq!(info.bytes_of("workspace"), workspace);
            let buffers = select_buffers(true, output);
            for rows in SELECT_ROWS {
                let planned = SelectEntry::Weighted(&entry).planned(rows).unwrap();
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    legacy_select(
                        (entry.symbol(), entry.merge_symbol()),
                        [entry.params(), entry.merge_params()],
                        planned,
                        rows,
                        true,
                        cp
                    ),
                    "{mode:?} at {rows} rows"
                );
            }
            // The head-sum selection.
            let prepared = backend
                .indexer_select(&head_sum_op(mode), 1024, rank, Need::rows(2048))
                .unwrap();
            let entry = head_sum_entry(head_sum::Kind::Select { ranks });
            assert_eq!(prepared.info().symbols[0], (entry.symbol(), 45_840));
            let buffers = select_buffers(false, output);
            for rows in SELECT_ROWS {
                let planned = SelectEntry::HeadSum(&entry).planned(rows).unwrap();
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    legacy_select(
                        (entry.symbol(), entry.merge_symbol()),
                        [entry.params(), entry.merge_params()],
                        planned,
                        rows,
                        false,
                        cp
                    ),
                    "head-sum {mode:?} at {rows} rows"
                );
            }
        }
    }

    /// The writer table: the one-split form at one split when registered,
    /// the radix merge (else the tree merge) at two or more, else the
    /// entry's merge.
    #[test]
    fn writers_follow_the_splits_and_the_registered_forms() {
        for one_split in [false, true] {
            for tree in [false, true] {
                for radix in [false, true] {
                    let one = if one_split {
                        Writer::OneSplit
                    } else {
                        Writer::Merge
                    };
                    let many = match (radix, tree) {
                        (true, _) => Writer::Radix,
                        (false, true) => Writer::Tree,
                        (false, false) => Writer::Merge,
                    };
                    assert_eq!(writer(1, one_split, tree, radix), one);
                    for splits in [2, 4, 64] {
                        assert_eq!(writer(splits, one_split, tree, radix), many);
                    }
                }
            }
        }
    }

    /// The weighted engine with every form a builder may run: the one-row
    /// one-split forms, the tree merges and the global radix merge.
    fn with_forms() -> Sm120 {
        let selects = entries(PROFILE, geometry());
        backend()
            .with_indexer_one_split(rows::entries(&selects, 1))
            .with_indexer_merge_tree(merge_tree::entries(&selects))
            .with_indexer_merge_radix(merge_radix::entries(&selects))
    }

    /// What a builder launches for selection `entry` at `rows` rows with
    /// `one_split`, `tree` and `radix` forms of the same symbol's entry:
    /// the one-split form alone at one split; else the split kernel, then
    /// at two or more splits the radix merge, else the tree merge, and at
    /// one split the entry's merge.
    fn builder_select(
        entry: &Entry,
        forms: (&[rows::Entry], &[merge_tree::Entry], &[merge_radix::Entry]),
        rows: u32,
        cp: [u32; 2],
    ) -> Vec<Launch> {
        let splits = entry.splits(rows);
        let mut named = vec![
            ("q", Argument::DeviceBuffer(1)),
            ("head_weights", Argument::DeviceBuffer(2)),
            ("row_batch", Argument::DeviceBuffer(6)),
            ("positions", Argument::DeviceBuffer(7)),
            ("live_rows", Argument::DeviceBuffer(8)),
            ("workspace", Argument::DeviceBuffer(9)),
            ("output", Argument::DeviceBuffer(10)),
            ("top_k", Argument::U32(1024)),
            ("splits", Argument::U32(splits)),
            ("cp_rank", Argument::U32(cp[0])),
            ("cp_size", Argument::U32(cp[1])),
        ];
        named.extend(page_arguments());
        let (one_split, trees, radixes) = forms;
        let of = |symbol: String| symbol == entry.symbol();
        if splits == 1
            && let Some(form) = one_split.iter().find(|f| of(f.select.symbol()))
        {
            return vec![launch(
                form.symbol(),
                rows,
                form.plan(rows).unwrap(),
                by_name(&form.params(), &named),
            )];
        }
        let select = launch(
            entry.symbol(),
            rows,
            entry.plan(rows).unwrap(),
            by_name(&entry.params(), &named),
        );
        let radix = radixes.iter().find(|f| splits > 1 && of(f.select.symbol()));
        let tree = trees.iter().find(|f| splits > 1 && of(f.select.symbol()));
        let (symbol, plan, params) = match (radix, tree) {
            (Some(form), _) => (form.symbol(), form.plan(rows).unwrap(), form.params()),
            (None, Some(form)) => (form.symbol(), form.plan(rows).unwrap(), form.params()),
            (None, None) => (
                entry.merge_symbol().unwrap(),
                entry.merge_plan(rows).unwrap().unwrap(),
                entry.merge_params(),
            ),
        };
        vec![select, launch(symbol, rows, plan, by_name(&params, &named))]
    }

    /// Each weighted selection with the forms registered launches what the
    /// builder launches at every row count up to 2048: the global one its
    /// radix merges and one-split form, the per-rank ones their tree merges
    /// and one-split forms at every group rank. The op keeps the entry's
    /// implementation, lists the symbols in the order the row counts first
    /// launch them and sizes the workspace of the rows that split.
    #[test]
    fn selections_run_their_forms_as_the_builder_does() {
        let selects = entries(PROFILE, geometry());
        let one_split = rows::entries(&selects, 1);
        let trees = merge_tree::entries(&selects);
        let radixes = merge_radix::entries(&selects);
        let forms = (&one_split[..], &trees[..], &radixes[..]);
        for (mode, cp, output) in [
            (IndexerSelectionMode::Global, [0, 1], 2051 * 4),
            (IndexerSelectionMode::PerRank { ranks: 2 }, [1, 2], 512 * 8),
            (IndexerSelectionMode::PerRank { ranks: 4 }, [0, 4], 512 * 8),
            (IndexerSelectionMode::PerRank { ranks: 4 }, [3, 4], 512 * 8),
        ] {
            let ranks = match mode {
                IndexerSelectionMode::Global => None,
                IndexerSelectionMode::PerRank { ranks } => Some(ranks),
            };
            let entry = entry(Kind::Select { ranks });
            let prepared = with_forms()
                .indexer_select(&select_op(mode), 1024, cp[0], Need::rows(2048))
                .unwrap();
            let plain = backend()
                .indexer_select(&select_op(mode), 1024, cp[0], Need::rows(2048))
                .unwrap();
            let info = prepared.info();
            assert_eq!(info.implementation, plain.info().implementation);
            let one = rows::Entry::of(&entry, 1).unwrap();
            let merge = match ranks {
                None => merge_radix::Entry::of(&entry).unwrap().symbol(),
                Some(_) => merge_tree::Entry::of(&entry).unwrap().symbol(),
            };
            // 1 to 375 rows split (into 64 down to 2); from 376 rows one
            // split runs the one-split form.
            assert_eq!(entry.splits(375), 2);
            assert_eq!(entry.splits(376), 1);
            assert_eq!(
                info.symbols,
                [
                    (entry.symbol(), 58_256),
                    (merge, 0),
                    (one.symbol(), one.dynamic_shared_bytes())
                ]
            );
            assert_eq!(
                info.bytes_of("workspace"),
                (1..376).map(|rows| entry.workspace_bytes(rows)).max()
            );
            let buffers = select_buffers(true, output);
            for rows in 1..=2048 {
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    builder_select(&entry, forms, rows, cp),
                    "{mode:?} rank {} at {rows} rows",
                    cp[0]
                );
            }
        }
    }

    /// With only some forms registered the entry's own kernels run where
    /// no form serves: its merge at one split without a one-split form, at
    /// two or more without a tree or radix merge.
    #[test]
    fn selections_run_their_own_kernels_where_no_form_serves() {
        let selects = entries(PROFILE, geometry());
        let trees = merge_tree::entries(&selects);
        let one_split = rows::entries(&selects, 1);
        let global = select_op(IndexerSelectionMode::Global);
        let entry = entry(Kind::Select { ranks: None });
        let buffers = select_buffers(true, 2051 * 4);
        for (registered, forms) in [
            (
                backend().with_indexer_merge_tree(trees.clone()),
                (&[][..], &trees[..], &[][..]),
            ),
            (
                backend().with_indexer_one_split(one_split.clone()),
                (&one_split[..], &[][..], &[][..]),
            ),
        ] {
            let prepared = registered
                .indexer_select(&global, 1024, 0, Need::rows(1024))
                .unwrap();
            for rows in [1, 2, 64, 188, 375, 376, 512, 1024] {
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    builder_select(&entry, forms, rows, [0, 1]),
                    "{rows} rows"
                );
            }
        }
        // Under one split's rows only, a one-split form is never launched
        // and not listed.
        let small = backend()
            .with_indexer_one_split(one_split.clone())
            .indexer_select(&global, 1024, 0, Need::rows(375))
            .unwrap();
        assert_eq!(
            small.info().symbols,
            [(entry.symbol(), 58_256), (entry.merge_symbol().unwrap(), 0)]
        );
        // Head-sum selections have no forms: their ops are unchanged.
        let head_sum = head_sum_op(IndexerSelectionMode::PerRank { ranks: 2 });
        let with = with_forms()
            .indexer_select(&head_sum, 1024, 1, Need::rows(2048))
            .unwrap();
        let without = backend()
            .indexer_select(&head_sum, 1024, 1, Need::rows(2048))
            .unwrap();
        assert_eq!(with.info(), without.info());
        let buffers = select_buffers(false, 512 * 8);
        for rows in [1, 32, 376, 2048] {
            assert_eq!(
                with.launches(&buffers, View::EMPTY, Counts::rows(rows)),
                without.launches(&buffers, View::EMPTY, Counts::rows(rows))
            );
        }
    }

    /// Forms serve the entry they were made from in every registration
    /// order; equal forms are one, and two forms of one entry that launch
    /// differently are a tie the op refuses by name.
    #[test]
    fn selection_forms_serve_their_entry_in_every_order() {
        let selects = entries(PROFILE, geometry());
        let global = entry(Kind::Select { ranks: None });
        let one = rows::Entry::of(&global, 1).unwrap();
        let two = rows::Entry::of(&global, 2).unwrap();
        let tree = merge_tree::Entry::of(&global).unwrap();
        let radix = merge_radix::Entry::of(&global).unwrap();
        // Forms of a twin entry of the same query under another name.
        let twin = Entry {
            name: format!("{}_twin", global.name),
            ..global.clone()
        };
        let twin_one = rows::Entry::of(&twin, 1).unwrap();
        let twin_radix = merge_radix::Entry::of(&twin).unwrap();
        #[derive(Clone, Debug)]
        enum Form {
            OneSplit(rows::Entry),
            Tree(merge_tree::Entry),
            Radix(merge_radix::Entry),
        }
        let register =
            |forms: &[Form]| {
                forms.iter().cloned().fold(
                    Sm120::new().with_pooled_indexer(selects.clone()),
                    |b, f| match f {
                        Form::OneSplit(f) => b.with_indexer_one_split([f]),
                        Form::Tree(f) => b.with_indexer_merge_tree([f]),
                        Form::Radix(f) => b.with_indexer_merge_radix([f]),
                    },
                )
            };
        let op = select_op(IndexerSelectionMode::Global);
        let symbols = |backend: Sm120| {
            backend
                .indexer_select(&op, 1024, 0, Need::rows(512))
                .map(|prepared| {
                    prepared
                        .info()
                        .symbols
                        .iter()
                        .map(|(symbol, _)| symbol.clone())
                        .collect::<Vec<_>>()
                })
        };
        let all = [
            Form::OneSplit(twin_one.clone()),
            Form::Radix(twin_radix.clone()),
            Form::OneSplit(one.clone()),
            Form::Tree(tree.clone()),
            Form::Radix(radix.clone()),
        ];
        for order in crate::backend::select::orders(&all) {
            assert_eq!(
                symbols(register(&order)),
                Ok(vec![global.symbol(), radix.symbol(), one.symbol()])
            );
        }
        assert_eq!(
            symbols(register(&all[..2])),
            Ok(vec![global.symbol(), global.merge_symbol().unwrap()])
        );
        assert_eq!(
            symbols(register(&[
                Form::Tree(tree.clone()),
                Form::Tree(tree.clone())
            ])),
            Ok(vec![
                global.symbol(),
                tree.symbol(),
                global.merge_symbol().unwrap()
            ])
        );
        for order in crate::backend::select::orders(&[Form::OneSplit(one), Form::OneSplit(two)]) {
            let Err(Refused::Rule(rule)) = symbols(register(&order)) else {
                panic!("two one-split forms were prepared");
            };
            assert!(
                rule.starts_with("two SM120 one-split pooled indexer selection entries"),
                "{rule}"
            );
        }
    }

    /// Equal forms made from entries of different capacities: a form
    /// plans through its own entry, so the op takes the least capacity that
    /// holds its rows in every registration order, and refuses when every
    /// matching form holds fewer rows.
    #[test]
    fn selection_forms_fit_the_rows_in_every_order() {
        let selects = entries(PROFILE, geometry());
        let global = entry(Kind::Select { ranks: None });
        let narrow = Entry {
            capacity: 376,
            ..global.clone()
        };
        let wide = rows::Entry::of(&global, 1).unwrap();
        let short = rows::Entry::of(&narrow, 1).unwrap();
        // The forms launch the same code under one symbol.
        assert_eq!(wide.symbol(), short.symbol());
        assert_eq!(
            format!("{:?}", wide.entry_spec()),
            format!("{:?}", short.entry_spec())
        );
        let op = select_op(IndexerSelectionMode::Global);
        let buffers = select_buffers(true, 2051 * 4);
        let backend = |forms: &[rows::Entry]| {
            Sm120::new()
                .with_pooled_indexer(selects.clone())
                .with_indexer_one_split(forms.to_vec())
        };
        let alone = backend(std::slice::from_ref(&wide))
            .indexer_select(&op, 1024, 0, Need::rows(2048))
            .unwrap();
        for order in crate::backend::select::orders(&[short.clone(), wide.clone()]) {
            for capacity in [376, 1024, 2048] {
                let prepared = backend(&order)
                    .indexer_select(&op, 1024, 0, Need::rows(capacity))
                    .unwrap();
                assert_eq!(prepared.info().symbols.last().unwrap().0, wide.symbol());
                for rows in [376, capacity] {
                    assert_eq!(
                        prepared
                            .launches(&buffers, View::EMPTY, Counts::rows(rows))
                            .unwrap(),
                        alone
                            .launches(&buffers, View::EMPTY, Counts::rows(rows))
                            .unwrap(),
                        "{capacity} capacity at {rows} rows"
                    );
                }
            }
        }
        let Err(Refused::Rule(rule)) =
            backend(&[short]).indexer_select(&op, 1024, 0, Need::rows(2048))
        else {
            panic!("a form of 376 rows served 2048");
        };
        assert!(rule.contains("exceed every SM120 one-split"), "{rule}");
    }

    /// A selection run through forms guarantees what every kernel that
    /// writes its output declares: the selection's claims when each form
    /// declares them, nothing for row counts a form declaring nothing (a
    /// lab mutant) writes. A model's claims for these symbols change
    /// nothing: the entries declare their own.
    #[test]
    fn selections_through_forms_claim_what_every_writer_declares() {
        let claim = Invariance::any_rows(2048).with_ties(kernel_core::properties::Ties::LowerIndex);
        let selects = entries(PROFILE, geometry());
        let select = entry(Kind::Select { ranks: Some(2) });
        assert_eq!(select.invariance(), claim);
        let tree = merge_tree::Entry::of(&select).unwrap();
        let one = rows::Entry::of(&select, 1).unwrap();
        let mutant = |suffix: &str| merge_tree::Entry {
            suffix: suffix.into(),
            ..merge_tree::Entry::of(&select).unwrap()
        };
        let mutant_one = rows::Entry {
            suffix: "_mutant".into(),
            ..one.clone()
        };
        assert_eq!(tree.invariance(), claim);
        assert_eq!(one.invariance(), claim);
        assert_eq!(mutant("_mutant").invariance(), Invariance::NONE);
        let op = select_op(IndexerSelectionMode::PerRank { ranks: 2 });
        let backend = |tree: merge_tree::Entry, one: rows::Entry| {
            Sm120::new()
                .with_pooled_indexer(selects.clone())
                .with_indexer_merge_tree([tree])
                .with_indexer_one_split([one])
                .with_context_parallel_claims([(select.merge_symbol().unwrap(), Invariance::NONE)])
        };
        // Small row counts run the tree merge, from 376 the one-split form.
        for (tree, one, capacity, granted) in [
            (mutant("_mutant"), one.clone(), 4, false),
            (tree.clone(), one.clone(), 4, true),
            (tree.clone(), mutant_one.clone(), 512, false),
            (tree.clone(), one.clone(), 512, true),
        ] {
            let label = format!("{} {} at {capacity}", tree.symbol(), one.symbol());
            let backend = backend(tree, one);
            let prepared = backend.indexer_select(&op, 1024, 1, Need::rows(capacity));
            assert_eq!(
                prepared.unwrap().info().guarantees,
                if granted { claim } else { Invariance::NONE },
                "{label}"
            );
            let exact =
                backend.indexer_select(&op, 1024, 1, Need::rows(capacity).with_exact(claim));
            assert_eq!(exact.is_ok(), granted, "{label}");
        }
    }

    fn merge_buffers(positions: bool, output: u64) -> IndexerMergeBuffers<View> {
        IndexerMergeBuffers {
            pairs: view(1, 4 * ROWS * 512 * 8),
            rank_rows: 2048,
            positions: positions.then(|| view(2, ROWS * 4)),
            live_rows: view(3, 4),
            output: view(4, ROWS * output),
        }
    }

    #[test]
    fn merges_launch_what_the_builder_plans() {
        // The weighted engine's merges and the block-id merges; the
        // head-sum key list merges are a second entry for the same op.
        let backend = Sm120::new()
            .with_pooled_indexer(entries(PROFILE, geometry()))
            .with_block_merges(block_merge::entries(PROFILE, head_sum_geometry(), &[2, 4]));
        let named = [
            ("pairs", Argument::DeviceBuffer(1)),
            ("positions", Argument::DeviceBuffer(2)),
            ("live_rows", Argument::DeviceBuffer(3)),
            ("lists", Argument::DeviceBuffer(4)),
            ("blocks", Argument::DeviceBuffer(4)),
            ("top_k", Argument::U32(2048)),
            ("rank_rows", Argument::U32(2048)),
        ];
        for ranks in [2, 4] {
            // Key lists: the weighted engine's merge.
            let op = ContextParallelQuery::TopKMerge {
                ranks,
                group: 4,
                max_top_k: 2048,
                always_tail: true,
                capacity: 2048,
            };
            let prepared = backend.indexer_merge(&op, 2048, Need::rows(2048)).unwrap();
            let entry = entry(Kind::TopKMerge { ranks });
            assert_eq!(prepared.info().symbols, [(entry.symbol(), 0)]);
            let buffers = merge_buffers(true, 2051 * 4);
            for rows in SELECT_ROWS {
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    [launch(
                        entry.symbol(),
                        rows,
                        entry.plan(rows).unwrap(),
                        by_name(&entry.params(), &named)
                    )]
                );
            }
            // Block ids.
            let op = ContextParallelQuery::TopKMergeBlocks {
                ranks,
                group: 4,
                max_top_k: 2048,
                capacity: 2048,
            };
            let prepared = backend.indexer_merge(&op, 2048, Need::rows(2048)).unwrap();
            let entry = block_merge::entries(PROFILE, head_sum_geometry(), &[ranks])
                .pop()
                .unwrap();
            let buffers = merge_buffers(false, 512 * 4);
            for rows in SELECT_ROWS {
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    [launch(
                        entry.symbol(),
                        rows,
                        entry.plan(rows).unwrap(),
                        by_name(&entry.params(), &named)
                    )]
                );
            }
        }
        // A backend with only the head-sum engine merges key lists with its
        // merge.
        let head_sum_only =
            Sm120::new().with_head_sum_indexer(head_sum::entries(PROFILE, head_sum_geometry()));
        let op = ContextParallelQuery::TopKMerge {
            ranks: 2,
            group: 4,
            max_top_k: 2048,
            always_tail: true,
            capacity: 2048,
        };
        let prepared = head_sum_only
            .indexer_merge(&op, 2048, Need::rows(16))
            .unwrap();
        assert_eq!(
            prepared.info().symbols[0].0,
            head_sum_entry(head_sum::Kind::TopKMerge { ranks: 2 }).symbol()
        );
    }

    /// Equal-shape weighted and head-sum key list merges are two entries
    /// for one op: the backend refuses the pair, whichever registered
    /// first, rather than launch either by registration order.
    #[test]
    fn equal_shape_merges_of_both_engines_refuse_in_either_order() {
        let op = ContextParallelQuery::TopKMerge {
            ranks: 2,
            group: 4,
            max_top_k: 2048,
            always_tail: true,
            capacity: 2048,
        };
        let head_sum_first = Sm120::new()
            .with_head_sum_indexer(head_sum::entries(PROFILE, head_sum_geometry()))
            .with_pooled_indexer(entries(PROFILE, geometry()));
        for backend in [head_sum_first, backend()] {
            let Err(Refused::Rule(rule)) = backend.indexer_merge(&op, 2048, Need::rows(16)) else {
                panic!("an ambiguous merge was prepared");
            };
            assert!(rule.contains("two SM120 pooled indexer entries serve the op"));
        }
        // The same entry registered twice is one entry.
        let twice = Sm120::new()
            .with_pooled_indexer(entries(PROFILE, geometry()))
            .with_pooled_indexer(entries(PROFILE, geometry()));
        assert_eq!(
            twice
                .indexer_merge(&op, 2048, Need::rows(16))
                .unwrap()
                .info()
                .symbols[0]
                .0,
            entry(Kind::TopKMerge { ranks: 2 }).symbol()
        );
    }

    /// The least capacity that holds the rows is found over every
    /// matching entry before a tie is judged, so registration order never
    /// changes the choice: two 2048-row entries and a 32-row one serve 16
    /// rows from the 32-row entry and refuse 64 rows as a tie, in every
    /// order.
    #[test]
    fn entry_choice_does_not_depend_on_registration_order() {
        let op = PooledIndexerQuery::CompactSelection {
            width: 2051,
            capacity: 32,
        };
        let compact = |name: &str, capacity| Entry {
            name: format!("{}_{name}", PROFILE.tag()),
            capacity,
            ..entry(Kind::Compact)
        };
        let all = [
            compact("compact_a", 2048),
            compact("compact_b", 2048),
            compact("compact_c", 32),
        ];
        let small = all[2].symbol();
        let orders = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        for order in orders {
            let backend = Sm120::new().with_pooled_indexer(order.map(|i| all[i].clone()));
            let prepared = backend.indexer_compact(&op, Need::rows(16)).unwrap();
            assert_eq!(prepared.info().symbols[0].0, small, "{order:?}");
            let Err(Refused::Rule(rule)) = backend.indexer_compact(&op, Need::rows(64)) else {
                panic!("{order:?}: a tie was prepared");
            };
            assert_eq!(
                rule,
                format!(
                    "two SM120 pooled indexer entries serve the op, {} and {}",
                    all[0].symbol(),
                    all[1].symbol()
                ),
                "{order:?}"
            );
        }
    }

    #[test]
    fn compactions_launch_what_the_builder_plans() {
        let op = PooledIndexerQuery::CompactSelection {
            width: 2051,
            capacity: 32,
        };
        let prepared = backend().indexer_compact(&op, Need::rows(32)).unwrap();
        let entry = entry(Kind::Compact);
        let buffers = SelectionCompactBuffers {
            lists: view(1, 2048 * 2051 * 4),
            source: view(2, 32 * 4),
            live_rows: view(3, 4),
            output: view(4, 32 * 2051 * 4),
        };
        let named = [
            ("lists", Argument::DeviceBuffer(1)),
            ("source", Argument::DeviceBuffer(2)),
            ("live_rows", Argument::DeviceBuffer(3)),
            ("out", Argument::DeviceBuffer(4)),
        ];
        for rows in [1, 4, 32] {
            assert_eq!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                [launch(
                    entry.symbol(),
                    rows,
                    entry.plan(rows).unwrap(),
                    by_name(&entry.params(), &named)
                )]
            );
        }
    }

    #[test]
    fn prepare_refuses_unserved_over_limit_and_inexact_ops() {
        let backend = backend();
        let rule = |result: Result<_, Refused>| match result {
            Err(Refused::Rule(rule)) => rule,
            Err(other) => panic!("{other:?}"),
            Ok(_) => panic!("prepared"),
        };
        let global = select_op(IndexerSelectionMode::Global);
        // SM120's rank ceiling, which the contract leaves to a backend.
        let nine = select_op(IndexerSelectionMode::PerRank { ranks: 9 });
        assert_eq!(
            rule(
                backend
                    .indexer_select(&nine, 1024, 0, Need::rows(4))
                    .map(|_| ())
            ),
            "per-rank indexer selection needs 2 to 8 ranks"
        );
        // A variant no entry instantiates: eight ranks.
        let eight = select_op(IndexerSelectionMode::PerRank { ranks: 8 });
        assert_eq!(
            rule(
                backend
                    .indexer_select(&eight, 1024, 0, Need::rows(4))
                    .map(|_| ())
            ),
            "no SM120 pooled indexer entry serves the op"
        );
        // Shapes the templates do not instantiate.
        let PooledIndexerQuery::Select {
            group,
            max_top_k,
            always_tail,
            granularity,
            key_format,
            mode,
            page_size,
            capacity,
            ..
        } = global
        else {
            unreachable!()
        };
        let shape = |heads, head_dim| PooledIndexerQuery::Select {
            heads,
            head_dim,
            group,
            max_top_k,
            always_tail,
            granularity,
            key_format,
            mode,
            page_size,
            capacity,
        };
        let eight_heads = PooledIndexerQuery::HeadSumSelect {
            heads: 8,
            head_dim: 128,
            group,
            max_top_k,
            always_tail,
            mode,
            page_size,
            capacity,
        };
        for op in [shape(24, 128), shape(80, 128), shape(32, 64), eight_heads] {
            assert!(
                rule(
                    backend
                        .indexer_select(&op, 1024, 0, Need::rows(4))
                        .map(|_| ())
                )
                .contains("outside the SM120 engine"),
                "{op:?}"
            );
        }
        // More rows than any entry holds.
        assert!(
            rule(
                backend
                    .indexer_select(&global, 1024, 0, Need::rows(4096))
                    .map(|_| ())
            )
            .contains("exceed every SM120 pooled indexer entry's capacity")
        );
        // Exactness the entries do not declare: rows past the capacity;
        // their one-row bits and tie rule are granted.
        assert!(matches!(
            backend.indexer_select(
                &global,
                1024,
                0,
                Need::rows(4).with_exact(Invariance::one_row(4096))
            ),
            Err(Refused::Inexact { .. })
        ));
        assert!(
            backend
                .indexer_select(
                    &global,
                    1024,
                    0,
                    Need::rows(4).with_exact(
                        Invariance::one_row(4).with_ties(kernel_core::properties::Ties::LowerIndex)
                    )
                )
                .is_ok()
        );
        // Runtime top-k and ranks outside the entry.
        for top_k in [0, 1022, 2052] {
            assert!(
                backend
                    .indexer_select(&global, top_k, 0, Need::rows(4))
                    .is_err()
            );
        }
        assert!(
            backend
                .indexer_select(&global, 1024, 1, Need::rows(4))
                .is_err()
        );
        let per_rank = select_op(IndexerSelectionMode::PerRank { ranks: 2 });
        assert!(
            backend
                .indexer_select(&per_rank, 1024, 2, Need::rows(4))
                .is_err()
        );
        // A query of another op.
        assert!(backend.indexer_compact(&global, Need::rows(4)).is_err());
        assert!(
            backend
                .indexer_key_write(&global, 0, 1, Need::rows(4))
                .is_err()
        );
        // A key width the template does not instantiate.
        let PooledIndexerQuery::KeyWrite {
            pooling,
            key_norm_eps_bits,
            page_size,
            key_format,
            ..
        } = key_write_op(RecurrentPhase::Prefill, 32)
        else {
            unreachable!()
        };
        let narrow = PooledIndexerQuery::KeyWrite {
            pooling: GatedGroupPooling {
                width: 64,
                ..pooling
            },
            key_norm_eps_bits,
            page_size,
            key_format,
            phase: RecurrentPhase::Prefill,
            capacity: 32,
        };
        assert!(
            rule(
                backend
                    .indexer_key_write(&narrow, 0, 1, Need::rows(32))
                    .map(|_| ())
            )
            .contains("outside the SM120 engine")
        );
        // A backend without the family's entries serves none of its ops.
        assert!(
            Sm120::new()
                .indexer_select(&global, 1024, 0, Need::rows(4))
                .is_err()
        );
    }

    #[test]
    fn launches_check_every_view() {
        let backend = backend();
        let write = backend
            .indexer_key_write(
                &key_write_op(RecurrentPhase::Prefill, 2048),
                0,
                1,
                Need::rows(64),
            )
            .unwrap();
        let good = key_write_buffers(false);
        write
            .launches(&good, View::EMPTY, Counts::rows(64))
            .unwrap();
        for rows in [0, 65] {
            assert!(
                write
                    .launches(&good, View::EMPTY, Counts::rows(rows))
                    .is_err()
            );
        }
        let short = |mut buffers: IndexerKeyWriteBuffers<View>,
                     edit: &dyn Fn(&mut IndexerKeyWriteBuffers<View>)| {
            edit(&mut buffers);
            write
                .launches(&buffers, View::EMPTY, Counts::rows(64))
                .is_err()
        };
        assert!(short(good, &|b| b.key.bytes = 64 * 256 - 1));
        assert!(short(good, &|b| b.gate_stride = 64));
        assert!(short(good, &|b| b.position.bytes = 4 * 256 - 1));
        assert!(short(good, &|b| b.tail.state.bytes = 300 * 768 * 2 - 1));
        // Prefill and decode ignore checkpoint ids a host binds anyway.
        let mut bound = good;
        bound.tail.checkpoint_ids = Some(view(11, 4));
        assert_eq!(
            write
                .launches(&bound, View::EMPTY, Counts::rows(64))
                .unwrap(),
            write
                .launches(&good, View::EMPTY, Counts::rows(64))
                .unwrap()
        );
        assert!(short(good, &|b| b.pooled.page_stride = 16 * 128 * 2 - 1));
        assert!(short(good, &|b| {
            b.pooled.keys.bytes = u64::from(PAGES - 1) * PAGE_STRIDE + 16 * 128 * 2 - 1
        }));
        // A wider key stride reads further.
        let mut strided = good;
        strided.key_stride = 256;
        strided.key.bytes = 63 * 512 + 256 - 1;
        assert!(
            write
                .launches(&strided, View::EMPTY, Counts::rows(64))
                .is_err()
        );
        strided.key.bytes += 1;
        write
            .launches(&strided, View::EMPTY, Counts::rows(64))
            .unwrap();
        // A verify write needs its checkpoints.
        let verify = RecurrentPhase::Verify { max_rows: 6 };
        let write = backend
            .indexer_key_write(&key_write_op(verify, 192), 0, 1, Need::rows(192))
            .unwrap();
        assert!(write.launches(&good, View::EMPTY, Counts::rows(6)).is_err());

        let select = backend
            .indexer_select(
                &select_op(IndexerSelectionMode::Global),
                1024,
                0,
                Need::rows(2048),
            )
            .unwrap();
        let good = select_buffers(true, 2051 * 4);
        select
            .launches(&good, View::EMPTY, Counts::rows(32))
            .unwrap();
        let workspace = entry(Kind::Select { ranks: None }).workspace_bytes(32);
        let mut buffers = good;
        buffers.workspace.bytes = workspace - 1;
        assert!(
            select
                .launches(&buffers, View::EMPTY, Counts::rows(32))
                .is_err()
        );
        buffers.workspace.bytes = workspace;
        select
            .launches(&buffers, View::EMPTY, Counts::rows(32))
            .unwrap();
        buffers.output.bytes = 32 * 2051 * 4 - 1;
        assert!(
            select
                .launches(&buffers, View::EMPTY, Counts::rows(32))
                .is_err()
        );
        buffers = good;
        buffers.head_weights = None;
        assert!(
            select
                .launches(&buffers, View::EMPTY, Counts::rows(32))
                .is_err()
        );
        let head_sum = backend
            .indexer_select(
                &head_sum_op(IndexerSelectionMode::Global),
                1024,
                0,
                Need::rows(2048),
            )
            .unwrap();
        // A head-sum selection ignores head weights.
        let mut unweighted = good;
        unweighted.head_weights = None;
        assert_eq!(
            head_sum
                .launches(&good, View::EMPTY, Counts::rows(32))
                .unwrap(),
            head_sum
                .launches(&unweighted, View::EMPTY, Counts::rows(32))
                .unwrap()
        );

        let merge = backend
            .indexer_merge(
                &ContextParallelQuery::TopKMergeBlocks {
                    ranks: 2,
                    group: 4,
                    max_top_k: 2048,
                    capacity: 2048,
                },
                2048,
                Need::rows(2048),
            )
            .unwrap();
        let mut buffers = merge_buffers(false, 512 * 4);
        buffers.rank_rows = 32;
        merge
            .launches(&buffers, View::EMPTY, Counts::rows(32))
            .unwrap();
        assert!(
            merge
                .launches(&buffers, View::EMPTY, Counts::rows(33))
                .is_err()
        );
        // A block-id merge ignores positions.
        let mut positioned = buffers;
        positioned.positions = Some(view(2, 4));
        assert_eq!(
            merge
                .launches(&positioned, View::EMPTY, Counts::rows(32))
                .unwrap(),
            merge
                .launches(&buffers, View::EMPTY, Counts::rows(32))
                .unwrap()
        );
    }

    /// A change of the weighted entry field `name` alone.
    fn weighted_change(name: &str) -> Box<dyn Fn(&mut Entry)> {
        let other = Profile::synthetic(crate::ProfileNames {
            name: "pooled-other",
            tag: "pooled_other",
        });
        match name {
            "profile" => Box::new(move |e| e.profile = other),
            "name" => Box::new(|e| e.name.push_str("_twin")),
            "kind" => Box::new(|e| {
                e.kind = if e.kind == Kind::Compact {
                    Kind::TopKMerge { ranks: 2 }
                } else {
                    Kind::Compact
                }
            }),
            "capacity" => Box::new(|e| e.capacity *= 2),
            "heads" => Box::new(|e| e.geometry.heads = 16),
            "head_dim" => Box::new(|e| e.geometry.head_dim = 64),
            "group" => Box::new(|e| e.geometry.group = 8),
            "page_size" => Box::new(|e| e.geometry.page_size = 128),
            "max_top_k" => Box::new(|e| e.geometry.max_top_k = 4096),
            "always_tail" => Box::new(|e| e.geometry.always_tail = !e.geometry.always_tail),
            "position_dtype" => Box::new(|e| e.geometry.position_dtype = DType::F32),
            "key_norm_eps" => Box::new(|e| e.geometry.key_norm_eps = 1e-3),
            "verify_rows" => Box::new(|e| e.geometry.verify_rows = 5),
            other => panic!("no change of {other}"),
        }
    }

    /// Every field of a weighted entry and its geometry.
    fn weighted_names() -> Vec<&'static str> {
        use crate::backend::select::field_names;
        let mut names = field_names!(Entry {
            profile,
            name,
            geometry,
            kind,
            capacity
        });
        names.retain(|name| *name != "geometry");
        names.extend(field_names!(crate::pooled_indexer::Geometry {
            heads,
            head_dim,
            group,
            page_size,
            max_top_k,
            always_tail,
            position_dtype,
            key_norm_eps,
            verify_rows
        }));
        names
    }

    /// A change of the head-sum or block-merge field `name` alone.
    fn head_sum_change<E: HeadSumFields>(name: &str) -> Box<dyn Fn(&mut E)> {
        let other = Profile::synthetic(crate::ProfileNames {
            name: "pooled-other",
            tag: "pooled_other",
        });
        match name {
            "profile" => Box::new(move |e| *e.profile() = other),
            "name" => Box::new(|e| e.name().push_str("_twin")),
            "kind" => Box::new(|e| e.change_kind()),
            "ranks" => Box::new(|e| e.change_kind()),
            "capacity" => Box::new(|e| *e.capacity() *= 2),
            "heads" => Box::new(|e| e.geometry().heads = 8),
            "head_dim" => Box::new(|e| e.geometry().head_dim = 64),
            "group" => Box::new(|e| e.geometry().group = 8),
            "page_size" => Box::new(|e| e.geometry().page_size = 128),
            "max_top_k" => Box::new(|e| e.geometry().max_top_k = 4096),
            "always_tail" => Box::new(|e| e.geometry().always_tail = !e.geometry().always_tail),
            other => panic!("no change of {other}"),
        }
    }

    /// The fields the head-sum and block-merge entries share.
    trait HeadSumFields {
        fn profile(&mut self) -> &mut Profile;
        fn name(&mut self) -> &mut String;
        fn capacity(&mut self) -> &mut u32;
        fn geometry(&mut self) -> &mut head_sum::Geometry;
        fn change_kind(&mut self);
    }

    impl HeadSumFields for head_sum::Entry {
        fn profile(&mut self) -> &mut Profile {
            &mut self.profile
        }
        fn name(&mut self) -> &mut String {
            &mut self.name
        }
        fn capacity(&mut self) -> &mut u32 {
            &mut self.capacity
        }
        fn geometry(&mut self) -> &mut head_sum::Geometry {
            &mut self.geometry
        }
        fn change_kind(&mut self) {
            self.kind = match self.kind {
                head_sum::Kind::Select { .. } => head_sum::Kind::TopKMerge { ranks: 2 },
                head_sum::Kind::TopKMerge { .. } => head_sum::Kind::Select { ranks: None },
            }
        }
    }

    impl HeadSumFields for block_merge::Entry {
        fn profile(&mut self) -> &mut Profile {
            &mut self.profile
        }
        fn name(&mut self) -> &mut String {
            &mut self.name
        }
        fn capacity(&mut self) -> &mut u32 {
            &mut self.capacity
        }
        fn geometry(&mut self) -> &mut head_sum::Geometry {
            &mut self.geometry
        }
        fn change_kind(&mut self) {
            self.ranks = if self.ranks == 2 { 4 } else { 2 };
        }
    }

    fn head_sum_geometry_names() -> Vec<&'static str> {
        crate::backend::select::field_names!(head_sum::Geometry {
            heads,
            head_dim,
            group,
            page_size,
            max_top_k,
            always_tail
        })
    }

    /// The fields of `names`, each `Inert` unless `classes` lists it.
    fn classified<E: 'static>(
        names: &[&'static str],
        classes: &[(&'static str, crate::backend::select::Class)],
        change: impl Fn(&str) -> Box<dyn Fn(&mut E)>,
    ) -> Vec<crate::backend::select::Field<E>> {
        use crate::backend::select::Class;
        names
            .iter()
            .map(|&name| {
                let class = classes
                    .iter()
                    .find(|(listed, _)| *listed == name)
                    .map_or(Class::Inert, |&(_, class)| class);
                (name, class, change(name))
            })
            .collect()
    }

    /// The pooled indexer keys hold what each kind's kernel reads, every
    /// field of every entry type changed alone: the auditor's block-id
    /// merge twins in pages or tail, which `topk_merge_blocks` never reads,
    /// keep its key; so do the weighted merge's heads and pages.
    #[test]
    fn launch_keys_hold_what_pooled_entries_launch() {
        use crate::backend::select::{Class, field_names, keys_follow_launches};
        let code = |e: &Entry| format!("{:?}", e.entry_specs());
        let named = [
            ("name", Class::Launch),
            ("kind", Class::Selects),
            ("head_dim", Class::Selects),
        ];
        let pooled = |e: &Entry| Sm120::new().with_pooled_indexer([e.clone()]);
        // A decode key write.
        let mut classes = named.to_vec();
        classes.extend([
            ("group", Class::Selects),
            ("page_size", Class::Selects),
            ("position_dtype", Class::Selects),
            ("key_norm_eps", Class::Selects),
        ]);
        let buffers = key_write_buffers(false);
        let op = key_write_op(RecurrentPhase::Decode, 32);
        keys_follow_launches(
            &entry(Kind::KeyWrite(RecurrentPhase::Decode)),
            weighted_names(),
            classified(&weighted_names(), &classes, weighted_change),
            weighted_launch,
            |e| {
                let prepared = pooled(e).indexer_key_write(&op, 0, 1, Need::rows(32))?;
                let launches = [1, 5, 32].map(|rows| {
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((code(e), launches))
            },
        );
        // A global weighted selection: its query holds the geometry it reads.
        let mut classes = named.to_vec();
        classes.extend(
            ["heads", "group", "page_size", "max_top_k", "always_tail"]
                .map(|name| (name, Class::Selects)),
        );
        let buffers = select_buffers(true, 2051 * 4);
        let op = select_op(IndexerSelectionMode::Global);
        keys_follow_launches(
            &entry(Kind::Select { ranks: None }),
            weighted_names(),
            classified(&weighted_names(), &classes, weighted_change),
            weighted_launch,
            |e| {
                let prepared = pooled(e).indexer_select(&op, 1024, 0, Need::rows(64))?;
                let launches = [1, 64].map(|rows| {
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((code(e), launches))
            },
        );
        // The weighted key-list merge: its traits name heads and pages it
        // never reads.
        let merge = ContextParallelQuery::TopKMerge {
            ranks: 2,
            group: 4,
            max_top_k: 2048,
            always_tail: true,
            capacity: 2048,
        };
        let mut classes = named.to_vec();
        classes.extend([
            ("heads", Class::Unread),
            ("page_size", Class::Unread),
            ("group", Class::Selects),
            ("max_top_k", Class::Selects),
            ("always_tail", Class::Selects),
        ]);
        let buffers = merge_buffers(true, 2051 * 4);
        keys_follow_launches(
            &entry(Kind::TopKMerge { ranks: 2 }),
            weighted_names(),
            classified(&weighted_names(), &classes, weighted_change),
            weighted_launch,
            |e| {
                let prepared = pooled(e).indexer_merge(&merge, 2048, Need::rows(64))?;
                let launches = [1, 64].map(|rows| {
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((code(e), launches))
            },
        );
        // A compaction: its list width.
        let compact = PooledIndexerQuery::CompactSelection {
            width: 2051,
            capacity: 32,
        };
        let mut classes = named.to_vec();
        classes.extend(["group", "max_top_k", "always_tail"].map(|name| (name, Class::Selects)));
        let buffers = SelectionCompactBuffers {
            lists: view(1, 2048 * 2051 * 4),
            source: view(2, 32 * 4),
            live_rows: view(3, 4),
            output: view(4, 32 * 2051 * 4),
        };
        keys_follow_launches(
            &entry(Kind::Compact),
            weighted_names(),
            classified(&weighted_names(), &classes, weighted_change),
            weighted_launch,
            |e| {
                let prepared = pooled(e).indexer_compact(&compact, Need::rows(32))?;
                let launches = [1, 32].map(|rows| {
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((code(e), launches))
            },
        );
        // The head-sum selection and merge.
        let mut names = field_names!(head_sum::Entry {
            profile,
            name,
            geometry,
            kind,
            capacity
        });
        names.retain(|name| *name != "geometry");
        names.extend(head_sum_geometry_names());
        let head_sum = |e: &head_sum::Entry| Sm120::new().with_head_sum_indexer([e.clone()]);
        let head_code = |e: &head_sum::Entry| format!("{:?}", e.entry_specs());
        let mut classes = vec![("name", Class::Launch), ("kind", Class::Selects)];
        classes.extend(
            [
                "heads",
                "head_dim",
                "group",
                "page_size",
                "max_top_k",
                "always_tail",
            ]
            .map(|name| (name, Class::Selects)),
        );
        let buffers = select_buffers(false, 2051 * 4);
        let op = head_sum_op(IndexerSelectionMode::Global);
        keys_follow_launches(
            &head_sum_entry(head_sum::Kind::Select { ranks: None }),
            names.clone(),
            classified(&names, &classes, head_sum_change),
            head_sum_launch,
            |e| {
                let prepared = head_sum(e).indexer_select(&op, 1024, 0, Need::rows(64))?;
                let launches = [1, 64].map(|rows| {
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((head_code(e), launches))
            },
        );
        let mut classes = vec![
            ("name", Class::Launch),
            ("kind", Class::Selects),
            ("page_size", Class::Unread),
        ];
        classes.extend(
            ["heads", "head_dim", "group", "max_top_k", "always_tail"]
                .map(|name| (name, Class::Selects)),
        );
        let buffers = merge_buffers(true, 2051 * 4);
        keys_follow_launches(
            &head_sum_entry(head_sum::Kind::TopKMerge { ranks: 2 }),
            names.clone(),
            classified(&names, &classes, head_sum_change),
            head_sum_launch,
            |e| {
                let prepared = head_sum(e).indexer_merge(&merge, 2048, Need::rows(64))?;
                let launches = [1, 64].map(|rows| {
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((head_code(e), launches))
            },
        );
        // The block-id merge: the auditor's page and tail twins are one.
        let mut names = field_names!(block_merge::Entry {
            profile,
            name,
            geometry,
            ranks,
            capacity
        });
        names.retain(|name| *name != "geometry");
        names.extend(head_sum_geometry_names());
        let blocks = ContextParallelQuery::TopKMergeBlocks {
            ranks: 2,
            group: 4,
            max_top_k: 2048,
            capacity: 2048,
        };
        let mut classes = vec![
            ("name", Class::Launch),
            ("ranks", Class::Selects),
            ("page_size", Class::Unread),
            ("always_tail", Class::Unread),
        ];
        classes
            .extend(["heads", "head_dim", "group", "max_top_k"].map(|name| (name, Class::Selects)));
        let buffers = merge_buffers(false, 512 * 4);
        let base = block_merge::entries(PROFILE, head_sum_geometry(), &[2]).remove(0);
        keys_follow_launches(
            &base,
            names.clone(),
            classified(&names, &classes, head_sum_change),
            block_launch,
            |e| {
                let prepared = Sm120::new().with_block_merges([e.clone()]).indexer_merge(
                    &blocks,
                    2048,
                    Need::rows(64),
                )?;
                let launches = [1, 64].map(|rows| {
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((format!("{:?}", e.entry_spec()), launches))
            },
        );
    }

    /// The auditor's case: block-id merges differing only in the pages or
    /// tail their traits name are one entry in every order.
    #[test]
    fn block_merges_differing_in_unread_traits_are_one() {
        let base = block_merge::entries(PROFILE, head_sum_geometry(), &[2]).remove(0);
        let mut pages = base.clone();
        pages.geometry.page_size = 128;
        let mut tail = base.clone();
        tail.geometry.always_tail = !tail.geometry.always_tail;
        let op = ContextParallelQuery::TopKMergeBlocks {
            ranks: 2,
            group: 4,
            max_top_k: 2048,
            capacity: 2048,
        };
        for order in crate::backend::select::orders(&[base.clone(), pages, tail]) {
            let prepared = Sm120::new()
                .with_block_merges(order)
                .indexer_merge(&op, 2048, Need::rows(64))
                .unwrap();
            assert_eq!(prepared.info().symbols[0].0, base.symbol());
        }
    }

    /// Compactions differing only in geometry a compaction never reads (the
    /// selection's heads) or in capacity past the rows are one entry in
    /// every order.
    #[test]
    fn compactions_differing_in_unread_geometry_are_one() {
        let op = PooledIndexerQuery::CompactSelection {
            width: 2051,
            capacity: 32,
        };
        let compact = entry(Kind::Compact);
        let mut heads = compact.clone();
        heads.geometry.heads = 16;
        let wider = Entry {
            capacity: compact.capacity * 2,
            ..compact.clone()
        };
        for order in crate::backend::select::orders(&[compact.clone(), heads, wider]) {
            let prepared = Sm120::new()
                .with_pooled_indexer(order)
                .indexer_compact(&op, Need::rows(32))
                .unwrap();
            assert_eq!(prepared.info().symbols[0].0, compact.symbol());
        }
    }
}
