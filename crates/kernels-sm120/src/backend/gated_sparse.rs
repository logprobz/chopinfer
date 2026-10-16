//! Gated and sparse attention through the entries a model instantiates: the
//! gated rows (`attention::gated`), the block-sparse indexer and attention
//! (`attention::sparse`) and the context-parallel selected partials
//! (`selected_partials`). An op is served by the entry whose query equals
//! the op's apart from the capacity, whose layout is the op's, and, of those
//! that keep `need.exact`, has the least capacity that holds `need.capacity`
//! rows. Registration order never decides: two such entries of different
//! symbols are refused. Block scores also run as the use's phase says:
//! prefill rows through the prefill-shaped scores (rows per CTA, bits kept
//! over whole-block chunk splits), decode and verify rows through the
//! decode-shaped scores (one row per CTA column, one-row bits for exact
//! verify), as the catalog's users launch verify steps today; the by-run
//! verify scores are not chosen. Each launch plans the entry for the step's
//! rows, a graph bucket whose rows past the device live count return, as
//! the model's builders plan it; the attention's split count is the entry's
//! for those rows.

use kernel_build::EntryParam;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::contract::{
    Argument, ArgumentList, DType, GatedAttentionQuery, LaunchPlan, Refusal, SparseAttentionQuery,
    Validate,
};
use kernel_core::gated_sparse::{
    BlockKeys, BlockScoresBuffers, BlockScoresOp, BlockTopKBuffers, GatedKvWriteBuffers,
    GatedKvWriteOp, IndexQueryBuffers, KeyRings, KvCache, OutputGateBuffers, OutputGateOp,
    SelectedPartialsBuffers, SelectedPartialsOp, SparseAttentionBuffers, SparseAttentionOp,
    SparseKeyWriteBuffers, SparseKeyWriteOp,
};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::properties::Invariance;

use super::context_parallel::Claims;
use super::select::{self, Candidate, Chosen};
use super::{MAX_CAPACITY, check_plan};
use crate::attention::sparse::Kind;
use crate::attention::{gated, sparse};
use crate::selected_partials;

/// A model's gated and sparse attention entries, in the order prepare
/// tries them.
#[derive(Clone, Debug, Default)]
pub struct GatedSparseEntries {
    /// Gated rows (`gated::entries`).
    pub gated: Vec<gated::Entry>,
    /// Indexer and attention entries (`sparse::indexer_entries`,
    /// `sparse::attention_entries`).
    pub sparse: Vec<sparse::Entry>,
    /// Selected partials (`selected_partials::entries`).
    pub partials: Vec<selected_partials::Entry>,
}

/// The gated KV write of `op` through the selection's entry for it.
pub(super) fn gated_kv_write(
    entries: Option<&GatedSparseEntries>,
    op: &GatedKvWriteOp,
    need: Need,
) -> Result<Prepared<GatedKvWriteBuffers<View>>, Refused> {
    op.validate()?;
    gated::check_query(&op.query)?;
    let Chosen { entry, guarantees } = choose(
        "gated KV write",
        entries.into_iter().flat_map(|e| &e.gated).filter(|entry| {
            matches!(entry.query, GatedAttentionQuery::QkNormRopeKvWrite { .. })
                && gated_shape(entry.query) == gated_shape(op.query)
        }),
        need,
    )?;
    entry.contract()?;
    let GatedAttentionQuery::QkNormRopeKvWrite {
        page_size,
        rotary_dim,
        ..
    } = op.query
    else {
        unreachable!("a validated gated KV write holds a KV write query");
    };
    Ok(Box::new(GatedKvWrite {
        rows: GatedRows::new(entry, guarantees, need)?,
        op: *op,
        max_pages: op.max_context.div_ceil(page_size),
        rotary: rotary_dim,
    }))
}

/// The output gate of `op` through the selection's entry for it.
pub(super) fn output_gate(
    entries: Option<&GatedSparseEntries>,
    op: &OutputGateOp,
    need: Need,
) -> Result<Prepared<OutputGateBuffers<View>>, Refused> {
    op.validate()?;
    gated::check_query(&op.query)?;
    let GatedAttentionQuery::OutputGate {
        heads, head_dim, ..
    } = op.query
    else {
        unreachable!("a validated output gate holds an output gate query");
    };
    // The packed rows' KV heads are a template argument of the body.
    let body = format!(
        "chop::attention::gated::output_gate<{heads},{},{head_dim},{head_dim}>(attention,qkv,out,live_rows);",
        op.kv_heads
    );
    let Chosen { entry, guarantees } = choose(
        "output gate",
        entries.into_iter().flat_map(|e| &e.gated).filter(|entry| {
            gated_shape(entry.query) == gated_shape(op.query) && entry.body == body
        }),
        need,
    )?;
    entry.contract()?;
    Ok(Box::new(OutputGate {
        rows: GatedRows::new(entry, guarantees, need)?,
        heads,
        head_dim,
        kv_heads: op.kv_heads,
    }))
}

/// The sparse index query of `op` through the selection's entry for it.
pub(super) fn index_query(
    entries: Option<&GatedSparseEntries>,
    op: &SparseAttentionQuery,
    need: Need,
) -> Result<Prepared<IndexQueryBuffers<View>>, Refused> {
    op.validate()?;
    sparse::check_query(op)?;
    if !matches!(op, SparseAttentionQuery::IndexerQueryGemmaNormMrope { .. }) {
        return Err(Refused::Rule(
            "an index query normalizes by an index query".into(),
        ));
    }
    let Chosen { entry, guarantees } = choose_sparse(entries, "index query", need, |entry| {
        entry.kind == Kind::Query && sparse_shape(entry.query) == sparse_shape(*op)
    })?;
    Ok(Box::new(IndexQuery {
        sparse: SparseEntry::new(entry, guarantees, need, Vec::new())?,
    }))
}

/// The sparse key write of `op` through the selection's entry for it.
pub(super) fn key_write(
    entries: Option<&GatedSparseEntries>,
    op: &SparseKeyWriteOp,
    need: Need,
) -> Result<Prepared<SparseKeyWriteBuffers<View>>, Refused> {
    op.validate()?;
    sparse::check_query(&op.query)?;
    let Chosen { entry, guarantees } = choose_sparse(entries, "sparse key write", need, |entry| {
        entry.kind == Kind::KeyWrite
            && sparse_shape(entry.query) == sparse_shape(op.query)
            && entry.geometry.index_heads == op.index_heads
            && entry.geometry.page_size == op.page_size
            && entry.geometry.max_context == op.max_context
    })?;
    Ok(Box::new(KeyWrite {
        sparse: SparseEntry::new(entry, guarantees, need, Vec::new())?,
    }))
}

/// The block scores of `op` at `usage` through the selection's entry for
/// the use's phase.
pub(super) fn block_scores(
    entries: Option<&GatedSparseEntries>,
    op: &BlockScoresOp,
    usage: OperationUse,
    need: Need,
) -> Result<Prepared<BlockScoresBuffers<View>>, Refused> {
    op.validate()?;
    sparse::check_query(&op.query)?;
    if need.capacity != usage.capacity {
        return Err(Refused::Rule(
            "block scores' capacity is their use's capacity".into(),
        ));
    }
    let prefill = usage.phase == ExecutionPhase::Prefill;
    let Chosen { entry, guarantees } = choose_sparse(entries, "block scores", need, |entry| {
        scores_for(entry.kind, prefill)
            && sparse_shape(entry.query) == sparse_shape(op.query)
            && entry.geometry.block == op.block
            && entry.geometry.page_size == op.page_size
            && entry.geometry.max_context == op.max_context
    })?;
    let scores = scores_bytes(entry, need.capacity)?;
    Ok(Box::new(BlockScores {
        sparse: SparseEntry::new(entry, guarantees, need, vec![("scores".into(), scores)])?,
    }))
}

/// The top-k of `op` through the selection's entry for it.
pub(super) fn block_top_k(
    entries: Option<&GatedSparseEntries>,
    op: &SparseAttentionQuery,
    need: Need,
) -> Result<Prepared<BlockTopKBuffers<View>>, Refused> {
    op.validate()?;
    sparse::check_query(op)?;
    if !matches!(op, SparseAttentionQuery::BlockTopK { .. }) {
        return Err(Refused::Rule(
            "a block top-k selects by a top-k query".into(),
        ));
    }
    let Chosen { entry, guarantees } = choose_sparse(entries, "block top-k", need, |entry| {
        entry.kind == Kind::TopK && sparse_shape(entry.query) == sparse_shape(*op)
    })?;
    let scores = scores_bytes(entry, need.capacity)?;
    Ok(Box::new(BlockTopK {
        sparse: SparseEntry::new(entry, guarantees, need, vec![("scores".into(), scores)])?,
    }))
}

/// The sparse attention of `op` through the selection's entry for it.
pub(super) fn attention(
    entries: Option<&GatedSparseEntries>,
    op: &SparseAttentionOp,
    need: Need,
) -> Result<Prepared<SparseAttentionBuffers<View>>, Refused> {
    op.validate()?;
    sparse::check_query(&op.query)?;
    let Chosen { entry, guarantees } = choose_sparse(entries, "sparse attention", need, |entry| {
        entry.kind == Kind::Attention
            && sparse_shape(entry.query) == sparse_shape(op.query)
            && entry.geometry.max_context == op.max_context
    })?;
    let workspace = (1..=need.capacity)
        .map(|rows| entry.workspace_bytes(rows))
        .max()
        .unwrap_or(0);
    Ok(Box::new(Attention {
        sparse: SparseEntry::new(
            entry,
            guarantees,
            need,
            vec![("workspace".into(), workspace)],
        )?,
    }))
}

/// The selected partials of `op` at group rank `group_rank` of `group_size`
/// through the selection's entry for it.
pub(super) fn selected_partials(
    entries: Option<&GatedSparseEntries>,
    claims: &Claims,
    op: &SelectedPartialsOp,
    group_rank: u32,
    group_size: u32,
    need: Need,
) -> Result<Prepared<SelectedPartialsBuffers<View>>, Refused> {
    op.validate()?;
    // The partials entries carry no claims; the model declares them.
    let claimed = entries
        .into_iter()
        .flat_map(|e| &e.partials)
        .filter(|entry| partials_shape(entry.query()) == partials_shape(op.query))
        .map(|entry| Partial {
            entry,
            guarantees: claims.of(&[&entry.symbol()]),
        })
        .collect::<Vec<_>>();
    let Chosen {
        entry: Partial { entry, .. },
        guarantees,
    } = choose("selected partials", &claimed, need)?;
    entry.contract()?;
    entry.check_rank(group_rank, group_size)?;
    let plan = entry.plan_rows(need.capacity)?;
    check_plan(&plan)?;
    let digest = kernel_build::digest(format!("{entry:?}").as_bytes());
    let info = OpInfo::new(
        format!(
            "sm120.selected_partials.v1[{}]#{}",
            entry.symbol(),
            &digest[..16]
        ),
        guarantees,
        vec![(entry.symbol(), plan.dynamic_shared_bytes)],
    );
    need.check(&info)?;
    Ok(Box::new(SelectedPartials {
        entry: (*entry).clone(),
        capacity: need.capacity,
        max_pages: op.max_context.div_ceil(entry.geometry.page),
        scale: op.softmax_scale,
        group: [group_rank, group_size],
        info,
    }))
}

fn no_entry(op: &str) -> Refused {
    Refused::Rule(format!("no SM120 {op} entry serves the op"))
}

fn check_capacity(need: Need) -> Result<(), Refused> {
    if need.capacity == 0 || need.capacity > MAX_CAPACITY {
        return Err(Refused::Rule(format!(
            "an SM120 gated or sparse attention op serves 1 to {MAX_CAPACITY} rows"
        )));
    }
    Ok(())
}

/// An entry prepare may choose.
trait Choice {
    fn capacity(&self) -> u32;
    fn guarantees(&self) -> Invariance;
    fn symbol(&self) -> String;
    /// What the entry launches: its symbol and the instantiation and
    /// argument shape it launches with, without its profile or claims.
    fn launch(&self) -> String;
}

impl Choice for gated::Entry {
    fn capacity(&self) -> u32 {
        gated::Entry::capacity(self)
    }
    fn guarantees(&self) -> Invariance {
        self.invariance
    }
    fn symbol(&self) -> String {
        self.symbol.clone()
    }
    fn launch(&self) -> String {
        let gated::Entry {
            profile: _,    // filters catalogs; no launch reads it
            query,         // the instantiation; its shape selects the op
            symbol,        // the launched symbol
            params,        // the ABI
            body,          // the code
            invariance: _, // claims
        } = self;
        format!("{symbol} {query:?} {params:?} {body}")
    }
}

impl Choice for sparse::Entry {
    fn capacity(&self) -> u32 {
        sparse::Entry::capacity(self)
    }
    fn guarantees(&self) -> Invariance {
        self.invariance
    }
    fn symbol(&self) -> String {
        sparse::Entry::symbol(self)
    }
    /// The geometry reaches a launch through the kernel's traits, plans,
    /// arguments and context limit, by what each kind's template in
    /// `cuda/attention/sparse_indexer.cuh` reads of its `Indexer` traits
    /// (the traits name every indexer field, but each kernel reads only
    /// some): the query norm reads the heads, dimensions and the H and W
    /// M-RoPE sections, with the norm epsilon as an argument; the key write
    /// also the block, ring and page, with the page count; the block scores
    /// the heads, dimension, block, top blocks and page, with the page
    /// count and score stride; the top-k the block and top blocks, with the
    /// score stride. Every kind refuses contexts past its limit. The
    /// attention reads its heads, scale and paging. No kind reads theta
    /// (the cos/sin table's) or the T section (the rest of the pairs),
    /// which the query norm's and key write's contracts fix.
    fn launch(&self) -> String {
        let sparse::Entry {
            profile: _,    // filters catalogs; no launch reads it
            name: _,       // in the symbol
            geometry,      // by kind, below
            kind,          // selects the op
            query,         // the instantiation; its shape selects the op
            invariance: _, // claims
        } = self;
        let sparse::Geometry {
            index_heads,
            index_dim,
            rotary_dim,
            theta: _, // the cos/sin table's; no kernel reads it
            mrope,    // by kind, below; no kernel reads the T section
            eps,
            block,
            ring,
            top_blocks,
            max_context,
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            softmax_scale,
        } = *geometry;
        let sections = [mrope[1], mrope[2]];
        let read = match kind {
            Kind::Attention => format!(
                "{:?}",
                (
                    q_heads,
                    kv_heads,
                    head_dim,
                    softmax_scale.to_bits(),
                    block,
                    page_size,
                    top_blocks,
                    max_context,
                )
            ),
            Kind::Query => format!(
                "{:?}",
                (
                    index_heads,
                    index_dim,
                    rotary_dim,
                    sections,
                    eps.to_bits(),
                    max_context,
                )
            ),
            Kind::KeyWrite => format!(
                "{:?}",
                (
                    index_heads,
                    index_dim,
                    rotary_dim,
                    sections,
                    eps.to_bits(),
                    block,
                    ring,
                    page_size,
                    max_context,
                )
            ),
            Kind::Scores { .. } => format!(
                "{:?}",
                (
                    index_heads,
                    index_dim,
                    block,
                    top_blocks,
                    page_size,
                    max_context,
                )
            ),
            Kind::TopK => format!("{:?}", (block, top_blocks, max_context)),
        };
        format!("{} {read} {kind:?} {query:?}", sparse::Entry::symbol(self))
    }
}

/// A selected partials entry with the claims the model declares for it.
#[derive(Clone, Copy, Debug)]
struct Partial<'a> {
    entry: &'a selected_partials::Entry,
    guarantees: Invariance,
}

impl Choice for Partial<'_> {
    fn capacity(&self) -> u32 {
        self.entry.capacity
    }
    fn guarantees(&self) -> Invariance {
        self.guarantees
    }
    fn symbol(&self) -> String {
        self.entry.symbol()
    }
    fn launch(&self) -> String {
        let selected_partials::Entry {
            profile: _,  // filters catalogs; no launch reads it
            name: _,     // in the symbol
            geometry,    // the query's; selects the op
            kv,          // the query's; selects the op
            phase,       // the query's; selects the op
            capacity: _, // bounds the rows the selection settles
        } = self.entry;
        format!("{} {geometry:?} {kv:?} {phase:?}", self.entry.symbol())
    }
}

/// The entry of `candidates` (those serving the op's shape and use) that
/// serves the op at `need`, whatever order the selection registered them
/// in (`select::choose`): of the entries that hold `need.capacity` rows
/// and keep `need.exact`, the least capacity; entries that launch the same
/// are one, claiming what all of them declare, and two that launch
/// differently are a tie.
fn choose<'a, E: Choice + std::fmt::Debug + 'a>(
    op: &str,
    candidates: impl IntoIterator<Item = &'a E>,
    need: Need,
) -> Result<Chosen<&'a E>, Refused> {
    check_capacity(need)?;
    let candidates: Vec<&E> = candidates.into_iter().collect();
    if !candidates.is_empty() && candidates.iter().all(|e| e.capacity() < need.capacity) {
        return Err(Refused::Rule(format!(
            "no SM120 {op} entry holds {} rows",
            need.capacity
        )));
    }
    let candidates: Vec<_> = candidates
        .into_iter()
        .map(|entry| {
            Candidate::new(entry, entry.symbol(), entry.launch(), entry.guarantees())
                .capacity(entry.capacity())
        })
        .collect();
    if candidates.is_empty() {
        return Err(no_entry(op));
    }
    select::choose(op, candidates, need)
}

fn choose_sparse<'a>(
    entries: Option<&'a GatedSparseEntries>,
    op: &str,
    need: Need,
    serves: impl Fn(&sparse::Entry) -> bool,
) -> Result<Chosen<&'a sparse::Entry>, Refused> {
    let chosen = choose(
        op,
        entries
            .into_iter()
            .flat_map(|e| &e.sparse)
            .filter(|entry| serves(entry)),
        need,
    )?;
    chosen.entry.contract()?;
    Ok(chosen)
}

/// Whether scores of `kind` run a use's rows: prefill rows take the
/// prefill-shaped scores (several rows per CTA), decode and verify rows the
/// decode-shaped ones (one row per CTA column). The by-run verify scores
/// serve no use.
fn scores_for(kind: Kind, prefill: bool) -> bool {
    match kind {
        Kind::Scores {
            rows,
            by_run: false,
        } => (rows > 1) == prefill,
        _ => false,
    }
}

/// `query` apart from its capacity.
fn gated_shape(query: GatedAttentionQuery) -> GatedAttentionQuery {
    let mut query = query;
    match &mut query {
        GatedAttentionQuery::QkNormRopeKvWrite { capacity, .. }
        | GatedAttentionQuery::OutputGate { capacity, .. } => *capacity = 0,
    }
    query
}

/// `query` apart from its capacity.
fn sparse_shape(query: SparseAttentionQuery) -> SparseAttentionQuery {
    let mut query = query;
    match &mut query {
        SparseAttentionQuery::IndexerQueryGemmaNormMrope { capacity, .. }
        | SparseAttentionQuery::IndexerKeyWrite { capacity, .. }
        | SparseAttentionQuery::BlockScores { capacity, .. }
        | SparseAttentionQuery::BlockTopK { capacity, .. }
        | SparseAttentionQuery::SparseAttention { capacity, .. } => *capacity = 0,
    }
    query
}

/// `query` apart from its capacity; `None` for other context-parallel ops.
fn partials_shape(query: ContextParallelQuery) -> Option<ContextParallelQuery> {
    let mut query = query;
    match &mut query {
        ContextParallelQuery::SelectedAttentionPartials { capacity, .. } => *capacity = 0,
        _ => return None,
    }
    Some(query)
}

/// The scores of `rows` rows: F32 rows of the entry's score stride.
fn scores_bytes(entry: &sparse::Entry, rows: u32) -> Result<u64, Refused> {
    let stride = entry.geometry.score_stride(entry.geometry.max_context);
    bytes(&[u64::from(rows), u64::from(stride), 4]).map_err(Refused::from)
}

fn buffer(view: View) -> Argument {
    Argument::DeviceBuffer(view.handle)
}

/// The product of `factors` bytes, refused rather than wrapped.
fn bytes(factors: &[u64]) -> Result<u64, Refusal> {
    factors
        .iter()
        .try_fold(1u64, |product, &factor| product.checked_mul(factor))
        .ok_or(Refusal("gated or sparse attention bytes overflow"))
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
        return Err(Refusal("sparse attention argument ABI mismatch"));
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

/// Checks a paged cache's planes hold `slots` pages of `page` bytes of K
/// and of V at their strides. An FP8 cache needs finite positive scales.
pub(super) fn check_cache(cache: &KvCache<View>, page: u64, fp8: bool) -> Result<(), Refusal> {
    if cache.slots == 0 {
        return Err(Refusal("a KV cache holds at least one page"));
    }
    for (plane, stride) in [
        (cache.k, cache.k_slot_stride),
        (cache.v, cache.v_slot_stride),
    ] {
        if stride < page {
            return Err(Refusal("a KV cache page stride is below the page's bytes"));
        }
        plane.covers(slots_end(cache.slots, stride, page)?)?;
    }
    if fp8
        && [cache.k_scale, cache.v_scale]
            .iter()
            .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(Refusal("an FP8 KV cache needs finite positive scales"));
    }
    Ok(())
}

/// The end of the last of `slots` slots of `bytes` bytes, `stride` apart.
fn slots_end(slots: u32, stride: u64, bytes: u64) -> Result<u64, Refusal> {
    u64::from(slots - 1)
        .checked_mul(stride)
        .and_then(|start| start.checked_add(bytes))
        .ok_or(Refusal("cache bytes overflow"))
}

/// Checks the block-key plane holds `keys.slots` pages of `page` bytes.
pub(super) fn check_block_keys(keys: &BlockKeys<View>, page: u64) -> Result<(), Refusal> {
    if keys.slots == 0 || keys.page_stride < page {
        return Err(Refusal(
            "a block-key plane holds pages of at least its block keys' bytes",
        ));
    }
    keys.keys
        .covers(slots_end(keys.slots, keys.page_stride, page)?)
}

/// Checks rings hold `rings.slots` slots of `ring` keys of `dim` and their
/// positions.
pub(super) fn check_rings(rings: &KeyRings<View>, ring: u32, dim: u32) -> Result<(), Refusal> {
    if rings.slots == 0 {
        return Err(Refusal("raw-key rings hold at least one slot"));
    }
    let rows = u64::from(rings.slots) * u64::from(ring);
    rings.keys.covers(bytes(&[rows, u64::from(dim), 2])?)?;
    rings.positions.covers(bytes(&[rows, 3, 4])?)
}

/// A table of `rotary / 2` cos and sin pairs per position covers the
/// step's rotary positions (`Counts::positions`).
pub(super) fn check_table(table: View, counts: Counts, rotary: u32) -> Result<(), Refusal> {
    table.covers(bytes(&[
        u64::from(counts.positions.max(1)),
        u64::from(rotary),
        4,
    ])?)
}

fn scale_bits(scale: f32) -> Argument {
    Argument::U32(scale.to_bits())
}

/// A gated row entry prepared for a capacity.
struct GatedRows {
    entry: gated::Entry,
    capacity: u32,
    info: OpInfo,
}

impl GatedRows {
    /// `entry`, claiming `guarantees` (what its registrations declare).
    fn new(entry: &gated::Entry, guarantees: Invariance, need: Need) -> Result<Self, Refused> {
        check_plan(&entry.plan_rows(need.capacity)?)?;
        let digest = kernel_build::digest(format!("{entry:?}").as_bytes());
        let info = OpInfo::new(
            format!("sm120.gated.v1[{}]#{}", entry.symbol, &digest[..16]),
            guarantees,
            vec![(entry.symbol.clone(), 0)],
        );
        need.check(&info)?;
        Ok(Self {
            entry: entry.clone(),
            capacity: need.capacity,
            info,
        })
    }

    fn launch(&self, counts: Counts, values: Vec<Argument>) -> Result<Vec<Launch>, Refusal> {
        Ok(vec![launch(
            self.entry.symbol.clone(),
            counts.rows,
            self.entry.plan_rows(counts.rows)?,
            self.entry.pack_arguments(values)?,
        )])
    }
}

/// `qk_norm_rope_kv`: a CTA per row; its first warps norm and rotate the Q
/// heads, the next ones each a K head, written with its V row.
struct GatedKvWrite {
    rows: GatedRows,
    op: GatedKvWriteOp,
    max_pages: u32,
    rotary: u32,
}

impl PreparedOp<GatedKvWriteBuffers<View>> for GatedKvWrite {
    fn info(&self) -> &OpInfo {
        &self.rows.info
    }

    fn launches(
        &self,
        buffers: &GatedKvWriteBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.rows.capacity)?;
        let GatedAttentionQuery::QkNormRopeKvWrite {
            q_heads,
            kv_heads,
            head_dim,
            eps_bits,
            page_size,
            kv_dtype,
            ..
        } = self.op.query
        else {
            unreachable!("a prepared gated KV write holds a KV write query");
        };
        let (q_heads, kv_heads, dim) =
            (u64::from(q_heads), u64::from(kv_heads), u64::from(head_dim));
        buffers
            .qkv
            .covers(bytes(&[rows, 2 * (q_heads + kv_heads), dim, 2])?)?;
        buffers.q.covers(bytes(&[rows, q_heads, dim, 2])?)?;
        buffers.token_batch.covers(rows * 4)?;
        buffers.positions.covers(rows * 4)?;
        buffers.live_rows.covers(4)?;
        // The host bounds each row's item by the block table's rows.
        buffers
            .block_tables
            .covers(bytes(&[items(counts), u64::from(self.max_pages), 4])?)?;
        let fp8 = kv_dtype == DType::Fp8E4M3;
        let element = if fp8 { 1 } else { 2 };
        let cache = &buffers.cache;
        check_cache(
            cache,
            bytes(&[u64::from(page_size), kv_heads, dim, element])?,
            fp8,
        )?;
        let rope = &buffers.rope;
        let (Some(cos_sin), Some(q_norm), Some(k_norm), Some(rope_positions)) =
            (rope.cos_sin, rope.q_norm, rope.k_norm, rope.rope_positions)
        else {
            return Err(Refusal(
                "a gated KV write reads its norm weights, RoPE positions and table",
            ));
        };
        q_norm.covers(dim * 2)?;
        k_norm.covers(dim * 2)?;
        rope_positions.covers(rows * 12)?;
        // The host bounds RoPE positions below the table's rows.
        check_table(cos_sin, counts, self.rotary)?;
        self.rows.launch(
            counts,
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
                Argument::U32(eps_bits),
                scale_bits(cache.k_scale),
                scale_bits(cache.v_scale),
            ],
        )
    }
}

/// `output_gate`: a CTA per row.
struct OutputGate {
    rows: GatedRows,
    heads: u32,
    head_dim: u32,
    kv_heads: u32,
}

impl PreparedOp<OutputGateBuffers<View>> for OutputGate {
    fn info(&self) -> &OpInfo {
        &self.rows.info
    }

    fn launches(
        &self,
        buffers: &OutputGateBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.rows.capacity)?;
        let (heads, dim) = (u64::from(self.heads), u64::from(self.head_dim));
        let row_bytes = bytes(&[rows, heads, dim, 2])?;
        buffers.attention.covers(row_bytes)?;
        buffers.output.covers(row_bytes)?;
        buffers.qkv.covers(bytes(&[
            rows,
            2 * (heads + u64::from(self.kv_heads)),
            dim,
            2,
        ])?)?;
        buffers.live_rows.covers(4)?;
        self.rows.launch(
            counts,
            vec![
                buffer(buffers.attention),
                buffer(buffers.qkv),
                buffer(buffers.output),
                buffer(buffers.live_rows),
            ],
        )
    }
}

/// A QSA entry prepared for a capacity.
struct SparseEntry {
    entry: sparse::Entry,
    capacity: u32,
    info: OpInfo,
}

impl SparseEntry {
    /// Checks the entry's widest plan against SM120. Its merge launches
    /// when a prepared row count splits, so it is listed then.
    /// `entry`, claiming `guarantees` (what its registrations declare).
    fn new(
        entry: &sparse::Entry,
        guarantees: Invariance,
        need: Need,
        bytes: Vec<(String, u64)>,
    ) -> Result<Self, Refused> {
        let g = entry.geometry;
        check_plan(&entry.plan_checked(need.capacity, g.max_context)?)?;
        let mut symbols = vec![(entry.symbol(), entry.dynamic_shared_bytes())];
        if let Some(merge) = entry.merge_symbol()
            && (1..=need.capacity).any(|rows| entry.splits(rows) > 1)
        {
            symbols.push((merge, 0));
        }
        let digest = kernel_build::digest(format!("{entry:?}").as_bytes());
        let mut info = OpInfo::new(
            format!("sm120.sparse.v1[{}]#{}", entry.symbol(), &digest[..16]),
            guarantees,
            symbols,
        );
        info.bytes = bytes;
        need.check(&info)?;
        Ok(Self {
            entry: entry.clone(),
            capacity: need.capacity,
            info,
        })
    }

    /// The step's rows, refused outside the prepared capacity, and its
    /// context.
    fn counts(&self, counts: Counts) -> Result<(u64, u32), Refusal> {
        let rows = rows(counts, self.capacity)?;
        if counts.context == 0 || counts.context > self.entry.geometry.max_context {
            return Err(Refusal("the step's context exceeds the prepared context"));
        }
        Ok((rows, counts.context))
    }

    fn max_pages(&self) -> u32 {
        let g = self.entry.geometry;
        g.max_context.div_ceil(g.page_size)
    }

    fn score_stride(&self) -> u32 {
        let g = self.entry.geometry;
        g.score_stride(g.max_context)
    }

    /// The block-key plane's bytes of one page.
    fn block_page(&self) -> u64 {
        self.entry.geometry.block_page_bytes()
    }

    fn launch(&self, counts: Counts, values: Vec<Argument>) -> Result<Launch, Refusal> {
        Ok(launch(
            self.entry.symbol(),
            counts.rows,
            self.entry.plan_checked(counts.rows, counts.context)?,
            pack(&self.entry.params(), values)?,
        ))
    }
}

/// `indexer_query`: a CTA per row, a warp per head.
struct IndexQuery {
    sparse: SparseEntry,
}

impl PreparedOp<IndexQueryBuffers<View>> for IndexQuery {
    fn info(&self) -> &OpInfo {
        &self.sparse.info
    }

    fn launches(
        &self,
        buffers: &IndexQueryBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (rows, _) = self.sparse.counts(counts)?;
        let g = self.sparse.entry.geometry;
        let (heads, dim) = (u64::from(g.index_heads), u64::from(g.index_dim));
        buffers
            .projection
            .covers(bytes(&[rows, heads + 1, dim, 2])?)?;
        buffers.weight.covers(dim * 2)?;
        buffers.rope_positions.covers(rows * 12)?;
        check_table(buffers.cos_sin, counts, g.rotary_dim)?;
        buffers.live_rows.covers(4)?;
        buffers.q.covers(bytes(&[rows, heads, dim, 2])?)?;
        Ok(vec![self.sparse.launch(
            counts,
            vec![
                buffer(buffers.projection),
                buffer(buffers.weight),
                buffer(buffers.rope_positions),
                buffer(buffers.cos_sin),
                buffer(buffers.live_rows),
                buffer(buffers.q),
                Argument::U32(g.eps.to_bits()),
            ],
        )?])
    }
}

/// `indexer_key_write`: a warp per row; a sequence's first row writes its
/// ring.
struct KeyWrite {
    sparse: SparseEntry,
}

impl PreparedOp<SparseKeyWriteBuffers<View>> for KeyWrite {
    fn info(&self) -> &OpInfo {
        &self.sparse.info
    }

    fn launches(
        &self,
        buffers: &SparseKeyWriteBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (rows, _) = self.sparse.counts(counts)?;
        let g = self.sparse.entry.geometry;
        let (heads, dim) = (u64::from(g.index_heads), u64::from(g.index_dim));
        let max_pages = self.sparse.max_pages();
        buffers
            .projection
            .covers(bytes(&[rows, heads + 1, dim, 2])?)?;
        buffers.weight.covers(dim * 2)?;
        buffers.rope_positions.covers(rows * 12)?;
        check_table(buffers.cos_sin, counts, g.rotary_dim)?;
        buffers.token_batch.covers(rows * 4)?;
        buffers.positions.covers(rows * 4)?;
        buffers.live_rows.covers(4)?;
        // The host bounds each row's item by its per-item arrays and their
        // slots by the rings.
        let items = items(counts);
        for per_item in [buffers.slot_ids, buffers.anchor_slots, buffers.anchor_ends] {
            per_item.covers(items * 4)?;
        }
        check_rings(&buffers.rings, g.ring, g.index_dim)?;
        check_rings(&buffers.anchors, g.ring, g.index_dim)?;
        buffers
            .block_tables
            .covers(bytes(&[items, u64::from(max_pages), 4])?)?;
        let block_keys = &buffers.block_keys;
        check_block_keys(block_keys, self.sparse.block_page())?;
        Ok(vec![self.sparse.launch(
            counts,
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
                Argument::U32(max_pages),
                Argument::U64(block_keys.page_stride),
                Argument::U32(g.eps.to_bits()),
            ],
        )?])
    }
}

/// `block_scores`: decode-shaped entries take one row per CTA column over
/// a fixed grid, prefill-shaped ones tiles of rows by key chunks.
struct BlockScores {
    sparse: SparseEntry,
}

impl PreparedOp<BlockScoresBuffers<View>> for BlockScores {
    fn info(&self) -> &OpInfo {
        &self.sparse.info
    }

    fn launches(
        &self,
        buffers: &BlockScoresBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (rows, _) = self.sparse.counts(counts)?;
        let g = self.sparse.entry.geometry;
        let max_pages = self.sparse.max_pages();
        let stride = self.sparse.score_stride();
        buffers.q.covers(bytes(&[
            rows,
            u64::from(g.index_heads),
            u64::from(g.index_dim),
            2,
        ])?)?;
        check_block_keys(&buffers.block_keys, self.sparse.block_page())?;
        buffers
            .block_tables
            .covers(bytes(&[items(counts), u64::from(max_pages), 4])?)?;
        buffers.token_batch.covers(rows * 4)?;
        buffers.positions.covers(rows * 4)?;
        buffers.live_rows.covers(4)?;
        buffers
            .scores
            .covers(bytes(&[rows, u64::from(stride), 4])?)?;
        Ok(vec![self.sparse.launch(
            counts,
            vec![
                buffer(buffers.q),
                buffer(buffers.block_keys.keys),
                buffer(buffers.block_tables),
                buffer(buffers.token_batch),
                buffer(buffers.positions),
                buffer(buffers.live_rows),
                buffer(buffers.scores),
                Argument::U32(max_pages),
                Argument::U64(buffers.block_keys.page_stride),
                Argument::U32(stride),
            ],
        )?])
    }
}

/// `block_topk`: a CTA per row, a radix select over the row's scores.
struct BlockTopK {
    sparse: SparseEntry,
}

impl PreparedOp<BlockTopKBuffers<View>> for BlockTopK {
    fn info(&self) -> &OpInfo {
        &self.sparse.info
    }

    fn launches(
        &self,
        buffers: &BlockTopKBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (rows, _) = self.sparse.counts(counts)?;
        let g = self.sparse.entry.geometry;
        let stride = self.sparse.score_stride();
        buffers
            .scores
            .covers(bytes(&[rows, u64::from(stride), 4])?)?;
        buffers.positions.covers(rows * 4)?;
        buffers.live_rows.covers(4)?;
        buffers
            .selected
            .covers(bytes(&[rows, u64::from(g.top_blocks), 4])?)?;
        Ok(vec![self.sparse.launch(
            counts,
            vec![
                buffer(buffers.scores),
                buffer(buffers.positions),
                buffer(buffers.live_rows),
                buffer(buffers.selected),
                Argument::U32(stride),
            ],
        )?])
    }
}

/// `attention`: a CTA per row, KV head and split, then the split merge
/// when the step's rows split.
struct Attention {
    sparse: SparseEntry,
}

impl PreparedOp<SparseAttentionBuffers<View>> for Attention {
    fn info(&self) -> &OpInfo {
        &self.sparse.info
    }

    fn launches(
        &self,
        buffers: &SparseAttentionBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (rows, _) = self.sparse.counts(counts)?;
        let entry = &self.sparse.entry;
        let SparseAttentionQuery::SparseAttention { kv_dtype, .. } = entry.query else {
            unreachable!("a prepared sparse attention holds an attention query");
        };
        let g = entry.geometry;
        let max_pages = self.sparse.max_pages();
        let dim = u64::from(g.head_dim);
        let row_bytes = bytes(&[rows, u64::from(g.q_heads), dim, 2])?;
        buffers.q.covers(row_bytes)?;
        buffers.output.covers(row_bytes)?;
        let fp8 = kv_dtype == DType::Fp8E4M3;
        let element = if fp8 { 1 } else { 2 };
        let cache = &buffers.cache;
        check_cache(
            cache,
            bytes(&[u64::from(g.page_size), u64::from(g.kv_heads), dim, element])?,
            fp8,
        )?;
        buffers
            .block_tables
            .covers(bytes(&[items(counts), u64::from(max_pages), 4])?)?;
        buffers.token_batch.covers(rows * 4)?;
        buffers.positions.covers(rows * 4)?;
        buffers.live_rows.covers(4)?;
        buffers
            .selected
            .covers(bytes(&[rows, u64::from(g.top_blocks), 4])?)?;
        buffers
            .workspace
            .covers(entry.workspace_bytes(counts.rows))?;
        let splits = entry.splits(counts.rows);
        let mut launches = vec![self.sparse.launch(
            counts,
            vec![
                buffer(buffers.q),
                buffer(cache.k),
                buffer(cache.v),
                buffer(buffers.block_tables),
                buffer(buffers.token_batch),
                buffer(buffers.positions),
                buffer(buffers.live_rows),
                buffer(buffers.selected),
                buffer(buffers.workspace),
                buffer(buffers.output),
                Argument::U32(max_pages),
                Argument::U64(cache.k_slot_stride),
                Argument::U64(cache.v_slot_stride),
                Argument::U32(splits),
                scale_bits(g.softmax_scale),
                scale_bits(cache.k_scale),
                scale_bits(cache.v_scale),
            ],
        )?];
        if let (Some(plan), Some(symbol)) = (entry.merge_plan(counts.rows)?, entry.merge_symbol()) {
            launches.push(launch(
                symbol,
                counts.rows,
                plan,
                pack(
                    &sparse::Entry::merge_params(),
                    vec![
                        buffer(buffers.workspace),
                        buffer(buffers.live_rows),
                        buffer(buffers.output),
                        Argument::U32(splits),
                    ],
                )?,
            ));
        }
        Ok(launches)
    }
}

/// `selected_partials`: a CTA per row and KV head.
struct SelectedPartials {
    entry: selected_partials::Entry,
    capacity: u32,
    max_pages: u32,
    scale: f32,
    group: [u32; 2],
    info: OpInfo,
}

impl PreparedOp<SelectedPartialsBuffers<View>> for SelectedPartials {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &SelectedPartialsBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let g = self.entry.geometry;
        let dim = u64::from(g.head_dim);
        let heads = u64::from(self.entry.heads());
        buffers.q.covers(bytes(&[rows, heads, dim, 2])?)?;
        let fp8 = self.entry.kv == DType::Fp8E4M3;
        let element = if fp8 { 1 } else { 2 };
        let cache = &buffers.cache;
        check_cache(
            cache,
            bytes(&[u64::from(g.page), u64::from(g.kv_heads), dim, element])?,
            fp8,
        )?;
        buffers
            .block_tables
            .covers(bytes(&[items(counts), u64::from(self.max_pages), 4])?)?;
        buffers.token_batch.covers(rows * 4)?;
        buffers.positions.covers(rows * 4)?;
        buffers.live_rows.covers(4)?;
        buffers
            .selected
            .covers(bytes(&[rows, u64::from(g.top_blocks), 4])?)?;
        let pairs = bytes(&[rows, heads, 4])?;
        buffers.max.covers(pairs)?;
        buffers.den.covers(pairs)?;
        buffers.num.covers(bytes(&[pairs, dim])?)?;
        let [group_rank, group_size] = self.group;
        Ok(vec![launch(
            self.entry.symbol(),
            counts.rows,
            self.entry.plan_rows(counts.rows)?,
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
                    buffer(buffers.selected),
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

#[cfg(test)]
mod tests {
    use super::super::{Sm120, view};
    use super::*;
    use crate::{Profile, ProfileNames};
    use kernel_core::backend::Backend;
    use kernel_core::gated_sparse::FoldInputs;
    use kernel_core::properties::{Chunk, Rows, Ties};

    const BIG: u64 = 1 << 40;
    const EXACT_ROWS: u32 = 4;
    const PAGE: u32 = 64;
    const STRIDE: u64 = 1 << 20;

    /// The shapes a selection serves: two rank shapes (a profile, its Q
    /// and KV heads per rank), and the indexer every rank shares.
    struct Shapes {
        ranks: [(Profile, u32, u32); 2],
        index_heads: u32,
        ring: u32,
        max_context: u32,
    }

    /// Served rank shapes: 6 and 12 Q heads over one KV head of 256, a
    /// four-head indexer whose prefill scores take 64-row tiles, a ring of
    /// 16 and a 262,144-token context of 65,536 blocks, the top-k bitmask's
    /// bound.
    const SERVED: Shapes = Shapes {
        ranks: [
            (
                Profile::synthetic(ProfileNames {
                    name: "gated-sparse-tp4",
                    tag: "gs_tp4",
                }),
                6,
                1,
            ),
            (
                Profile::synthetic(ProfileNames {
                    name: "gated-sparse-tp2",
                    tag: "gs_tp2",
                }),
                12,
                1,
            ),
        ],
        index_heads: 4,
        ring: 16,
        max_context: 262_144,
    };

    /// Synthetic shapes: 16 Q heads over two KV heads and 8 over one, a
    /// 16-head indexer, a ring of 8 and a 16,384-token context.
    const SYNTHETIC: Shapes = Shapes {
        ranks: [
            (
                Profile::synthetic(ProfileNames {
                    name: "gated-sparse-wide",
                    tag: "gs_wide",
                }),
                16,
                2,
            ),
            (
                Profile::synthetic(ProfileNames {
                    name: "gated-sparse-narrow",
                    tag: "gs_narrow",
                }),
                8,
                1,
            ),
        ],
        index_heads: 16,
        ring: 8,
        max_context: 16_384,
    };

    const SHAPES: [&Shapes; 2] = [&SERVED, &SYNTHETIC];

    impl Shapes {
        fn max_pages(&self) -> u32 {
            self.max_context / PAGE
        }

        /// The gated rows of `q_heads` Q heads over `kv_heads` KV heads of
        /// 256, M-RoPE over 64 dimensions and 64-token pages.
        fn config(&self, q_heads: u32, kv_heads: u32) -> gated::Config {
            gated::Config {
                q_heads,
                kv_heads,
                head_dim: 256,
                rotary: 64,
                theta: 1e7,
                sections: [11, 11, 10],
                eps: 1e-6,
                page: PAGE,
                max_rows: 8192,
            }
        }

        /// Block-sparse attention of `q_heads` Q heads over one KV head of
        /// 256, blocks of four and 512 selected blocks.
        fn geometry(&self, q_heads: u32) -> sparse::Geometry {
            sparse::Geometry {
                index_heads: self.index_heads,
                index_dim: 128,
                rotary_dim: 64,
                theta: 1e7,
                mrope: [11, 11, 10],
                eps: 1e-6,
                block: 4,
                ring: self.ring,
                top_blocks: 512,
                max_context: self.max_context,
                q_heads,
                kv_heads: 1,
                head_dim: 256,
                page_size: PAGE,
                softmax_scale: 0.0625,
            }
        }

        /// Each rank shape's gated rows and attention, the indexer once (it
        /// serves every rank), and the selected partials of the first rank
        /// shape's Q heads over one of its KV heads.
        fn selection(&self) -> GatedSparseEntries {
            let mut entries = GatedSparseEntries::default();
            let (first, q_heads, kv_heads) = self.ranks[0];
            entries.sparse.extend(sparse::indexer_entries(
                first,
                self.geometry(q_heads),
                EXACT_ROWS,
            ));
            for (profile, q_heads, kv_heads) in self.ranks {
                entries
                    .gated
                    .extend(gated::entries(profile, self.config(q_heads, kv_heads)));
                entries.sparse.extend(sparse::attention_entries(
                    profile,
                    self.geometry(q_heads),
                    EXACT_ROWS,
                ));
            }
            entries.partials = selected_partials::entries(
                first,
                selected_partials::Geometry {
                    q_heads: q_heads / kv_heads,
                    kv_heads: 1,
                    head_dim: 256,
                    page: PAGE,
                    block: 4,
                    top_blocks: 512,
                },
            );
            entries
        }

        fn backend(&self) -> Sm120 {
            Sm120::new().with_gated_sparse(self.selection())
        }

        /// The indexer entry `name`.
        fn indexer(&self, name: &str) -> sparse::Entry {
            self.entry(self.ranks[0].0, name)
        }

        /// Rank shape `rank`'s attention entry `name`.
        fn attention(&self, rank: usize, name: &str) -> sparse::Entry {
            self.entry(self.ranks[rank].0, name)
        }

        fn entry(&self, profile: Profile, name: &str) -> sparse::Entry {
            let name = format!("{}_{name}", profile.tag());
            self.selection()
                .sparse
                .into_iter()
                .find(|entry| entry.name == name)
                .unwrap()
        }

        fn kv_write_op(&self, q_heads: u32, kv_heads: u32, kv_dtype: DType) -> GatedKvWriteOp {
            let entry = gated::entries(self.ranks[0].0, self.config(q_heads, kv_heads))
                .into_iter()
                .find(|entry| {
                    matches!(entry.query,
                    GatedAttentionQuery::QkNormRopeKvWrite { kv_dtype: dtype, .. } if dtype == kv_dtype)
                })
                .unwrap();
            GatedKvWriteOp {
                query: entry.query,
                max_context: self.max_context,
            }
        }

        fn scores_op(&self) -> BlockScoresOp {
            BlockScoresOp {
                query: SparseAttentionQuery::BlockScores {
                    heads: self.index_heads,
                    head_dim: 128,
                    max_blocks: self.max_context / 4,
                    capacity: 1,
                },
                block: 4,
                page_size: PAGE,
                max_context: self.max_context,
            }
        }
    }

    fn cache(fp8: bool) -> KvCache<View> {
        KvCache {
            k: view(20, BIG),
            v: view(21, BIG),
            slots: 64,
            k_slot_stride: STRIDE,
            v_slot_stride: STRIDE,
            k_scale: if fp8 { 0.5 } else { 1.0 },
            v_scale: if fp8 { 0.25 } else { 1.0 },
        }
    }

    fn scales(fp8: bool) -> [Argument; 2] {
        let [k, v] = if fp8 { [0.5_f32, 0.25] } else { [1.0, 1.0] };
        [Argument::U32(k.to_bits()), Argument::U32(v.to_bits())]
    }

    fn ptr(handle: u64) -> Argument {
        Argument::DeviceBuffer(handle)
    }

    fn use_of(phase: ExecutionPhase, capacity: u32) -> OperationUse {
        OperationUse {
            phase,
            capacity,
            graph: phase != ExecutionPhase::Prefill,
        }
    }

    fn one_row() -> Invariance {
        Invariance {
            rows: Rows::OneRow { max: EXACT_ROWS },
            ..Invariance::NONE
        }
    }

    fn gated_buffers(fp8: bool) -> GatedKvWriteBuffers<View> {
        GatedKvWriteBuffers {
            qkv: view(1, BIG),
            q: view(2, BIG),
            cache: cache(fp8),
            block_tables: view(3, BIG),
            token_batch: view(4, BIG),
            positions: view(5, BIG),
            live_rows: view(6, 4),
            rope: FoldInputs {
                cos_sin: Some(view(7, BIG)),
                q_norm: Some(view(8, 512)),
                k_norm: Some(view(9, 512)),
                rope_positions: Some(view(10, BIG)),
            },
        }
    }

    fn block_keys() -> BlockKeys<View> {
        BlockKeys {
            keys: view(30, BIG),
            slots: 64,
            page_stride: STRIDE,
        }
    }

    fn scores_buffers() -> BlockScoresBuffers<View> {
        BlockScoresBuffers {
            q: view(1, BIG),
            block_keys: block_keys(),
            block_tables: view(2, BIG),
            token_batch: view(3, BIG),
            positions: view(4, BIG),
            live_rows: view(5, 4),
            scores: view(6, BIG),
        }
    }

    /// The legacy launch of a QSA entry: its plan for `rows` and `context`
    /// and `values` in parameter order.
    fn qsa(entry: &sparse::Entry, rows: u32, context: u32, values: Vec<Argument>) -> Launch {
        launch(
            entry.symbol(),
            rows,
            entry.plan(rows, context).unwrap(),
            ArgumentList(values),
        )
    }

    /// The legacy block scores launch of `entry` over [`scores_buffers`].
    fn legacy_scores(s: &Shapes, entry: &sparse::Entry, rows: u32, context: u32) -> Launch {
        let g = entry.geometry;
        qsa(
            entry,
            rows,
            context,
            vec![
                ptr(1),
                ptr(30),
                ptr(2),
                ptr(3),
                ptr(4),
                ptr(5),
                ptr(6),
                Argument::U32(s.max_pages()),
                Argument::U64(STRIDE),
                Argument::U32(g.score_stride(g.max_context)),
            ],
        )
    }

    #[test]
    fn gated_rows_launch_what_the_builder_plans() {
        for s in SHAPES {
            let backend = s.backend();
            let entries = s.selection().gated;
            for (_, q_heads, kv_heads) in s.ranks {
                for (dtype, fp8) in [(DType::Bf16, false), (DType::Fp8E4M3, true)] {
                    let op = s.kv_write_op(q_heads, kv_heads, dtype);
                    let prepared = backend.gated_kv_write(&op, Need::rows(8192)).unwrap();
                    let entry = entries
                        .iter()
                        .find(|entry| gated_shape(entry.query) == gated_shape(op.query))
                        .unwrap();
                    assert_eq!(prepared.info().guarantees, entry.invariance);
                    assert_eq!(prepared.info().symbols, [(entry.symbol.clone(), 0)]);
                    let buffers = gated_buffers(fp8);
                    let [k_scale, v_scale] = scales(fp8);
                    for rows in [1, 7, 32, 256, 8192] {
                        let launches = prepared
                            .launches(
                                &buffers,
                                View::EMPTY,
                                Counts::rows(rows)
                                    .with_items(rows, 4096)
                                    .with_positions(4096),
                            )
                            .unwrap();
                        let legacy = vec![
                            ptr(1),
                            ptr(8),
                            ptr(9),
                            ptr(10),
                            ptr(7),
                            ptr(4),
                            ptr(5),
                            ptr(6),
                            ptr(3),
                            ptr(20),
                            ptr(21),
                            ptr(2),
                            Argument::U32(s.max_pages()),
                            Argument::U64(STRIDE),
                            Argument::U64(STRIDE),
                            Argument::U32(1e-6_f32.to_bits()),
                            k_scale.clone(),
                            v_scale.clone(),
                        ];
                        assert_eq!(
                            launches,
                            [launch(
                                entry.symbol.clone(),
                                rows,
                                entry.plan(rows).unwrap(),
                                entry.pack_arguments(legacy).unwrap(),
                            )]
                        );
                    }
                }
                let gate = GatedAttentionQuery::OutputGate {
                    heads: q_heads,
                    head_dim: 256,
                    capacity: 8192,
                };
                let op = OutputGateOp {
                    query: gate,
                    kv_heads,
                };
                let prepared = backend.output_gate(&op, Need::rows(4096)).unwrap();
                let entry = entries
                    .iter()
                    .find(|entry| {
                        entry.query == gate
                            && entry.body.contains(&format!("<{q_heads},{kv_heads},"))
                    })
                    .unwrap();
                let buffers = OutputGateBuffers {
                    attention: view(1, BIG),
                    qkv: view(2, BIG),
                    output: view(3, BIG),
                    live_rows: view(4, 4),
                };
                for rows in [1, 16, 4096] {
                    assert_eq!(
                        prepared
                            .launches(&buffers, View::EMPTY, Counts::rows(rows))
                            .unwrap(),
                        [launch(
                            entry.symbol.clone(),
                            rows,
                            entry.plan(rows).unwrap(),
                            entry
                                .pack_arguments(vec![ptr(1), ptr(2), ptr(3), ptr(4)])
                                .unwrap(),
                        )]
                    );
                }
                assert!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(4097))
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn indexer_ops_launch_what_the_builders_plan() {
        for s in SHAPES {
            let backend = s.backend();
            let g = s.geometry(s.ranks[0].1);
            let eps = Argument::U32(g.eps.to_bits());
            let max_pages = Argument::U32(s.max_pages());
            let stride = g.score_stride(g.max_context);
            let contexts = [1, 300, 4096, s.max_context];
            // The index query.
            let query = s.indexer("index_query");
            let prepared = backend
                .sparse_index_query(&query.query, Need::rows(8192))
                .unwrap();
            assert_eq!(prepared.info().guarantees, query.invariance);
            let buffers = IndexQueryBuffers {
                projection: view(1, BIG),
                weight: view(2, 256),
                rope_positions: view(3, BIG),
                cos_sin: view(4, BIG),
                live_rows: view(5, 4),
                q: view(6, BIG),
            };
            for (rows, context) in [(1, s.max_context), (64, 300), (8192, 8192)] {
                assert_eq!(
                    prepared
                        .launches(
                            &buffers,
                            View::EMPTY,
                            Counts::rows(rows)
                                .with_items(1, context)
                                .with_positions(context)
                        )
                        .unwrap(),
                    [qsa(
                        &query,
                        rows,
                        context,
                        vec![ptr(1), ptr(2), ptr(3), ptr(4), ptr(5), ptr(6), eps.clone()]
                    )]
                );
            }
            // The key write.
            let write = s.indexer("index_key_write");
            let op = SparseKeyWriteOp {
                query: write.query,
                index_heads: s.index_heads,
                page_size: PAGE,
                max_context: s.max_context,
            };
            let prepared = backend.sparse_key_write(&op, Need::rows(8192)).unwrap();
            assert_eq!(prepared.info().guarantees, write.invariance);
            // The spare anchor ring is complete only past a ring of rows: a
            // need of anchors at any span (four live rows anchored at their
            // end) is refused, one past the ring granted.
            let anchors = |anchors| Invariance {
                anchors,
                ..Invariance::NONE
            };
            let spare = kernel_core::properties::Anchors::Spare;
            let past = kernel_core::properties::Anchors::SpareAfter { rows: g.ring };
            assert!(matches!(
                backend.sparse_key_write(&op, Need::rows(4).with_exact(anchors(spare))),
                Err(Refused::Inexact { .. })
            ));
            assert!(
                backend
                    .sparse_key_write(&op, Need::rows(4).with_exact(anchors(past)))
                    .is_ok()
            );
            let rings = |handle| KeyRings {
                keys: view(handle, BIG),
                positions: view(handle + 1, BIG),
                slots: 32,
            };
            let buffers = SparseKeyWriteBuffers {
                projection: view(1, BIG),
                weight: view(2, 256),
                rope_positions: view(3, BIG),
                cos_sin: view(4, BIG),
                token_batch: view(5, BIG),
                positions: view(6, BIG),
                live_rows: view(7, 4),
                slot_ids: view(8, BIG),
                rings: rings(9),
                anchor_slots: view(11, BIG),
                anchor_ends: view(12, BIG),
                anchors: rings(13),
                block_tables: view(15, BIG),
                block_keys: block_keys(),
            };
            for (rows, context) in [(1, s.max_context), (33, 2048), (8192, 8192)] {
                assert_eq!(
                    prepared
                        .launches(
                            &buffers,
                            View::EMPTY,
                            Counts::rows(rows)
                                .with_items(4, context)
                                .with_positions(context)
                        )
                        .unwrap(),
                    [qsa(
                        &write,
                        rows,
                        context,
                        vec![
                            ptr(1),
                            ptr(2),
                            ptr(3),
                            ptr(4),
                            ptr(5),
                            ptr(6),
                            ptr(7),
                            ptr(8),
                            ptr(9),
                            ptr(10),
                            ptr(11),
                            ptr(12),
                            ptr(13),
                            ptr(14),
                            ptr(15),
                            ptr(30),
                            max_pages.clone(),
                            Argument::U64(STRIDE),
                            eps.clone(),
                        ]
                    )]
                );
            }
            // A decode row at position 127 completes block 124..127, whose
            // first row's rotary position 100 an earlier step stored in the
            // ring: the step's rotary positions count it, though its own row
            // rotates at position 1, and the table must hold row 100.
            let SparseAttentionQuery::IndexerKeyWrite { rotary_dim, .. } = write.query else {
                unreachable!("a key write query");
            };
            let table = |positions: u64| SparseKeyWriteBuffers {
                cos_sin: view(4, positions * u64::from(rotary_dim) * 4),
                ..buffers
            };
            let completing = Counts::rows(1).with_items(1, 128).with_positions(101);
            assert!(
                prepared
                    .launches(&table(101), View::EMPTY, completing)
                    .is_ok()
            );
            assert!(
                prepared
                    .launches(&table(2), View::EMPTY, completing)
                    .is_err()
            );
            // Block scores: decode-shaped for decode and verify uses,
            // prefill-shaped (tiles of 64 rows at four indexer heads, 16 at
            // sixteen) for prefill uses.
            let tile = 64.min(256 / s.index_heads);
            for (usage, need, name, rows) in [
                (
                    use_of(ExecutionPhase::Decode, 256),
                    Need::rows(256).with_exact(one_row()),
                    "block_scores_decode",
                    vec![1, 4, 256],
                ),
                (
                    use_of(ExecutionPhase::Verify, 4),
                    Need::rows(4),
                    "block_scores_decode",
                    vec![1, 4],
                ),
                (
                    use_of(ExecutionPhase::Prefill, 2048),
                    Need::rows(2048).with_exact(Invariance {
                        chunk: Chunk::Granule(4),
                        ..Invariance::NONE
                    }),
                    "block_scores_prefill",
                    vec![1, tile - 1, tile, tile + 1, 300, 2048],
                ),
                (
                    use_of(ExecutionPhase::Prefill, 64),
                    Need::rows(64),
                    "block_scores_prefill",
                    vec![1, 17, 64],
                ),
            ] {
                let entry = s.indexer(name);
                if name == "block_scores_prefill" {
                    assert_eq!(
                        entry.kind,
                        Kind::Scores {
                            rows: tile,
                            by_run: false
                        }
                    );
                }
                let prepared = backend
                    .sparse_block_scores(&s.scores_op(), usage, need)
                    .unwrap();
                assert_eq!(prepared.info().guarantees, entry.invariance);
                assert_eq!(
                    prepared.info().bytes_of("scores"),
                    Some(u64::from(need.capacity) * u64::from(stride) * 4)
                );
                for rows in rows {
                    for context in contexts {
                        assert_eq!(
                            prepared
                                .launches(
                                    &scores_buffers(),
                                    View::EMPTY,
                                    Counts::rows(rows)
                                        .with_items(rows, context)
                                        .with_positions(context)
                                )
                                .unwrap(),
                            [legacy_scores(s, &entry, rows, context)]
                        );
                    }
                }
            }
            // The top-k, with the lower-index ties it declares.
            let top_k = s.indexer("block_topk");
            let need = Need::rows(2048).with_exact(Invariance {
                ties: Ties::LowerIndex,
                ..Invariance::NONE
            });
            let prepared = backend.sparse_block_top_k(&top_k.query, need).unwrap();
            let buffers = BlockTopKBuffers {
                scores: view(1, BIG),
                positions: view(2, BIG),
                live_rows: view(3, 4),
                selected: view(4, BIG),
            };
            for rows in [1, 9, 2048] {
                for context in contexts {
                    assert_eq!(
                        prepared
                            .launches(
                                &buffers,
                                View::EMPTY,
                                Counts::rows(rows)
                                    .with_items(1, context)
                                    .with_positions(context)
                            )
                            .unwrap(),
                        [qsa(
                            &top_k,
                            rows,
                            context,
                            vec![ptr(1), ptr(2), ptr(3), ptr(4), Argument::U32(stride)]
                        )]
                    );
                }
            }
        }
        // At the served context the top-k covers 65,536 blocks, its
        // bitmask's bound; one block more is refused.
        let top_k = SERVED.indexer("block_topk");
        assert_eq!(
            top_k.query,
            SparseAttentionQuery::BlockTopK {
                block: 4,
                top_blocks: 512,
                max_blocks: 65_536,
                capacity: 2048,
            }
        );
        let past = SparseAttentionQuery::BlockTopK {
            block: 4,
            top_blocks: 512,
            max_blocks: 65_537,
            capacity: 2048,
        };
        assert_eq!(past.validate(), Ok(()));
        assert!(matches!(
            SERVED.backend().sparse_block_top_k(&past, Need::rows(1)),
            Err(Refused::Rule(rule)) if rule.contains("outside the SM120 engine")
        ));
    }

    /// Every order of the three block scores entries, before and after the
    /// other entries in either order, chooses the same entry for each use.
    #[test]
    fn block_scores_choose_by_use_in_every_registration_order() {
        let s = &SERVED;
        let selection = s.selection();
        let (scores, others): (Vec<_>, Vec<_>) = selection
            .sparse
            .iter()
            .cloned()
            .partition(|entry| matches!(entry.kind, Kind::Scores { .. }));
        assert_eq!(scores.len(), 3);
        let decode = s.indexer("block_scores_decode");
        let prefill = s.indexer("block_scores_prefill");
        let uses = [
            (use_of(ExecutionPhase::Decode, 4), Need::rows(4), &decode),
            (use_of(ExecutionPhase::Verify, 4), Need::rows(4), &decode),
            (
                use_of(ExecutionPhase::Verify, 4),
                Need::rows(4).with_exact(one_row()),
                &decode,
            ),
            (use_of(ExecutionPhase::Prefill, 4), Need::rows(4), &prefill),
            (
                use_of(ExecutionPhase::Prefill, 2048),
                Need::rows(2048),
                &prefill,
            ),
        ];
        let orders = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut prepared_orders = 0;
        for order in orders {
            for scores_first in [true, false] {
                for reversed in [false, true] {
                    let mut rest = others.clone();
                    if reversed {
                        rest.reverse();
                    }
                    let ordered = order.iter().map(|&index| scores[index].clone());
                    let sparse = if scores_first {
                        ordered.chain(rest).collect()
                    } else {
                        rest.into_iter().chain(ordered).collect()
                    };
                    let backend = Sm120::new().with_gated_sparse(GatedSparseEntries {
                        sparse,
                        ..s.selection()
                    });
                    for &(usage, need, entry) in &uses {
                        let prepared = backend
                            .sparse_block_scores(&s.scores_op(), usage, need)
                            .unwrap();
                        let rows = need.capacity;
                        assert_eq!(
                            prepared
                                .launches(
                                    &scores_buffers(),
                                    View::EMPTY,
                                    Counts::rows(rows)
                                        .with_items(1, 16_384)
                                        .with_positions(16_384)
                                )
                                .unwrap(),
                            [legacy_scores(s, entry, rows, 16_384)],
                            "{order:?} {usage:?}"
                        );
                    }
                    prepared_orders += 1;
                }
            }
        }
        assert_eq!(prepared_orders, 24);
        // A use's capacity is the need's, and an exactness the use's
        // scores lack is refused, not served by the other shape.
        let backend = s.backend();
        assert!(matches!(
            backend.sparse_block_scores(
                &s.scores_op(),
                use_of(ExecutionPhase::Decode, 8),
                Need::rows(4)
            ),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            backend.sparse_block_scores(
                &s.scores_op(),
                use_of(ExecutionPhase::Prefill, 4),
                Need::rows(4).with_exact(one_row())
            ),
            Err(Refused::Inexact { .. })
        ));
        // Decode scores hold 256 rows; a wider graph step is refused.
        assert!(matches!(
            backend.sparse_block_scores(
                &s.scores_op(),
                use_of(ExecutionPhase::Decode, 257),
                Need::rows(257)
            ),
            Err(Refused::Rule(rule)) if rule.contains("holds 257 rows")
        ));
    }

    /// Two entries of different symbols that serve an op at the least
    /// capacity are refused in either order; equal entries are not a tie.
    #[test]
    fn ties_are_refused_whatever_the_order() {
        let s = &SERVED;
        let (first, q_heads, _) = s.ranks[0];
        let (second, _, _) = s.ranks[1];
        let query = s.indexer("index_query");
        for profiles in [[first, second], [second, first]] {
            let sparse = profiles
                .iter()
                .flat_map(|&profile| {
                    sparse::indexer_entries(profile, s.geometry(q_heads), EXACT_ROWS)
                })
                .collect();
            let backend = Sm120::new().with_gated_sparse(GatedSparseEntries {
                sparse,
                ..GatedSparseEntries::default()
            });
            assert!(matches!(
                backend.sparse_index_query(&query.query, Need::rows(1)),
                Err(Refused::Rule(rule)) if rule.contains("two SM120 index query entries")
            ));
        }
        let twice = GatedSparseEntries {
            sparse: [s.selection().sparse, s.selection().sparse].concat(),
            ..GatedSparseEntries::default()
        };
        assert!(
            Sm120::new()
                .with_gated_sparse(twice)
                .sparse_index_query(&query.query, Need::rows(1))
                .is_ok()
        );
    }

    /// An entry registered again with other claims launches the same, so
    /// it is one entry in every order: a need only the claiming one keeps
    /// takes it, and with no need the op claims what both declare.
    #[test]
    fn claims_of_one_launch_do_not_depend_on_order() {
        let s = &SHAPES[0];
        let entry = s.attention(0, "attn_bf16_decode");
        assert_ne!(entry.invariance, Invariance::NONE);
        let unclaimed = sparse::Entry {
            invariance: Invariance::NONE,
            ..entry.clone()
        };
        let op = SparseAttentionOp {
            query: entry.query,
            max_context: s.max_context,
        };
        let prepare = |need: Need| {
            let results = [
                vec![entry.clone(), unclaimed.clone()],
                vec![unclaimed.clone(), entry.clone()],
            ]
            .map(|sparse| {
                Sm120::new()
                    .with_gated_sparse(GatedSparseEntries {
                        sparse,
                        ..GatedSparseEntries::default()
                    })
                    .sparse_attention(&op, need)
                    .map(|prepared| prepared.info().guarantees)
            });
            assert_eq!(results[0], results[1]);
            results[0].clone()
        };
        let exact = Need::rows(256).with_exact(entry.invariance);
        assert_eq!(prepare(exact), Ok(entry.invariance));
        assert_eq!(
            prepare(Need::rows(256)),
            Ok(entry.invariance.meet(Invariance::NONE))
        );
    }

    /// Every field of a sparse entry and its geometry, listed exhaustively.
    fn sparse_names() -> Vec<&'static str> {
        use super::select::field_names;
        let mut names = field_names!(sparse::Entry {
            profile,
            name,
            geometry,
            kind,
            query,
            invariance
        });
        names.retain(|name| *name != "geometry");
        names.extend(field_names!(sparse::Geometry {
            index_heads,
            index_dim,
            rotary_dim,
            theta,
            mrope,
            eps,
            block,
            ring,
            top_blocks,
            max_context,
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            softmax_scale
        }));
        names
    }

    /// A change of the sparse entry field `name` alone, to a value its
    /// geometry still validates where one exists.
    fn sparse_change(name: &str) -> Box<dyn Fn(&mut sparse::Entry)> {
        let other = Profile::synthetic(ProfileNames {
            name: "gated-sparse-other",
            tag: "gs_other",
        });
        match name {
            "profile" => Box::new(move |e| e.profile = other),
            "name" => Box::new(|e| e.name.push_str("_twin")),
            "kind" => Box::new(|e| {
                e.kind = if e.kind == Kind::TopK {
                    Kind::Query
                } else {
                    Kind::TopK
                }
            }),
            "query" => Box::new(|e| {
                e.query = SERVED.attention(1, "attn_bf16_prefill").query;
                if e.kind == Kind::Attention {
                    e.query = SERVED.indexer("block_topk").query;
                }
            }),
            "invariance" => Box::new(|e| e.invariance = Invariance::NONE),
            "index_heads" => Box::new(|e| e.geometry.index_heads = 8),
            "index_dim" => Box::new(|e| e.geometry.index_dim = 64),
            "rotary_dim" => Box::new(|e| e.geometry.rotary_dim = 32),
            "theta" => Box::new(|e| e.geometry.theta = 1e6),
            "mrope" => Box::new(|e| e.geometry.mrope = [11, 10, 11]),
            "eps" => Box::new(|e| e.geometry.eps = 1e-5),
            "block" => Box::new(|e| e.geometry.block = 8),
            "ring" => Box::new(|e| e.geometry.ring = 32),
            "top_blocks" => Box::new(|e| e.geometry.top_blocks = 256),
            "max_context" => Box::new(|e| e.geometry.max_context = 16_384),
            "q_heads" => Box::new(|e| e.geometry.q_heads = 12),
            "kv_heads" => Box::new(|e| e.geometry.kv_heads = 2),
            "head_dim" => Box::new(|e| e.geometry.head_dim = 128),
            "page_size" => Box::new(|e| e.geometry.page_size = 32),
            "softmax_scale" => Box::new(|e| e.geometry.softmax_scale = 0.5),
            other => panic!("no change of {other}"),
        }
    }

    /// The fields of `classes` (unlisted ones `Inert`), each with its change.
    fn sparse_fields(
        classes: &[(&'static str, super::select::Class)],
    ) -> Vec<super::select::Field<sparse::Entry>> {
        use super::select::Class;
        sparse_names()
            .into_iter()
            .map(|name| {
                let class = classes
                    .iter()
                    .find(|(listed, _)| *listed == name)
                    .map_or(Class::Inert, |&(_, class)| class);
                (name, class, sparse_change(name))
            })
            .collect()
    }

    /// The code an entry's symbols run.
    fn sparse_code(entry: &sparse::Entry) -> String {
        format!("{:?}", entry.entry_specs())
    }

    /// The sparse key holds what each kind launches, every field of the
    /// entry and its geometry changed alone: each indexer kind the traits
    /// its kernel reads, its arguments and its context limit (a trait its
    /// kernel never reads changes only the traits' text: `Unread`); the
    /// attention its heads, scale and paging. The contract fixes the indexer's dimensions, so changing
    /// them alone serves no op. Launches run at contexts inside every
    /// changed limit.
    #[test]
    fn sparse_launch_keys_hold_what_each_kind_launches() {
        use super::select::{Class, keys_follow_launches};
        let s = &SERVED;
        let contexts = [1, 300, 8192];
        let backend = |entry: &sparse::Entry| {
            Sm120::new().with_gated_sparse(GatedSparseEntries {
                sparse: vec![entry.clone()],
                ..GatedSparseEntries::default()
            })
        };
        let at = |rows: u32| {
            contexts
                .iter()
                .map(move |&context| Counts::rows(rows).with_items(1, context))
        };
        // Fixed by every kind's contract.
        let fixed = [
            ("kind", Class::Selects),
            ("query", Class::Selects),
            ("index_dim", Class::Selects),
            ("rotary_dim", Class::Selects),
            ("name", Class::Launch),
        ];
        // The index query.
        let query = s.indexer("index_query");
        let buffers = IndexQueryBuffers {
            projection: view(1, BIG),
            weight: view(2, 256),
            rope_positions: view(3, BIG),
            cos_sin: view(4, BIG),
            live_rows: view(5, 4),
            q: view(6, BIG),
        };
        let mut classes = fixed.to_vec();
        classes.extend([
            ("index_heads", Class::Selects),
            ("theta", Class::Selects),
            ("mrope", Class::Selects),
            ("eps", Class::Selects),
            ("block", Class::Unread),
            ("ring", Class::Unread),
            ("top_blocks", Class::Unread),
            ("page_size", Class::Unread),
            ("max_context", Class::Bound),
        ]);
        keys_follow_launches(
            &query,
            sparse_names(),
            sparse_fields(&classes),
            Choice::launch,
            |e| {
                let op = backend(e).sparse_index_query(&query.query, Need::rows(8))?;
                let launches = at(8)
                    .map(|counts| op.launches(&buffers, View::EMPTY, counts).unwrap())
                    .collect::<Vec<_>>();
                Ok((sparse_code(e), launches))
            },
        );
        // The key write.
        let write = s.indexer("index_key_write");
        let rings = |handle| KeyRings {
            keys: view(handle, BIG),
            positions: view(handle + 1, BIG),
            slots: 32,
        };
        let buffers = SparseKeyWriteBuffers {
            projection: view(1, BIG),
            weight: view(2, 256),
            rope_positions: view(3, BIG),
            cos_sin: view(4, BIG),
            token_batch: view(5, BIG),
            positions: view(6, BIG),
            live_rows: view(7, 4),
            slot_ids: view(8, BIG),
            rings: rings(9),
            anchor_slots: view(11, BIG),
            anchor_ends: view(12, BIG),
            anchors: rings(13),
            block_tables: view(15, BIG),
            block_keys: block_keys(),
        };
        let op = SparseKeyWriteOp {
            query: write.query,
            index_heads: s.index_heads,
            page_size: PAGE,
            max_context: s.max_context,
        };
        let mut classes = fixed.to_vec();
        classes.extend([
            ("index_heads", Class::Selects),
            ("theta", Class::Selects),
            ("mrope", Class::Selects),
            ("eps", Class::Selects),
            ("block", Class::Selects),
            ("ring", Class::Selects),
            ("top_blocks", Class::Unread),
            ("page_size", Class::Selects),
            ("max_context", Class::Selects),
        ]);
        keys_follow_launches(
            &write,
            sparse_names(),
            sparse_fields(&classes),
            Choice::launch,
            |e| {
                let prepared = backend(e).sparse_key_write(&op, Need::rows(8))?;
                let launches = contexts
                    .iter()
                    .map(|&context| {
                        prepared
                            .launches(
                                &buffers,
                                View::EMPTY,
                                Counts::rows(8).with_items(4, context),
                            )
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                Ok((sparse_code(e), launches))
            },
        );
        // The decode block scores.
        let scores = s.indexer("block_scores_decode");
        let buffers = scores_buffers();
        let mut classes = fixed.to_vec();
        classes.extend([
            ("index_heads", Class::Selects),
            ("mrope", Class::Unread),
            ("block", Class::Selects),
            ("ring", Class::Unread),
            ("top_blocks", Class::Launch),
            ("page_size", Class::Selects),
            ("max_context", Class::Selects),
        ]);
        keys_follow_launches(
            &scores,
            sparse_names(),
            sparse_fields(&classes),
            Choice::launch,
            |e| {
                let op = backend(e).sparse_block_scores(
                    &s.scores_op(),
                    use_of(ExecutionPhase::Decode, 8),
                    Need::rows(8),
                )?;
                let launches = at(8)
                    .map(|counts| op.launches(&buffers, View::EMPTY, counts).unwrap())
                    .collect::<Vec<_>>();
                Ok((sparse_code(e), launches))
            },
        );
        // The block top-k: the auditor's theta and eps twins are one.
        let top_k = s.indexer("block_topk");
        let buffers = BlockTopKBuffers {
            scores: view(1, BIG),
            positions: view(2, BIG),
            live_rows: view(3, 4),
            selected: view(4, BIG),
        };
        let mut classes = fixed.to_vec();
        classes.extend([
            ("index_heads", Class::Unread),
            ("mrope", Class::Unread),
            ("block", Class::Selects),
            ("ring", Class::Unread),
            ("top_blocks", Class::Selects),
            ("page_size", Class::Unread),
            ("max_context", Class::Selects),
        ]);
        keys_follow_launches(
            &top_k,
            sparse_names(),
            sparse_fields(&classes),
            Choice::launch,
            |e| {
                let op = backend(e).sparse_block_top_k(&top_k.query, Need::rows(8))?;
                let launches = at(8)
                    .map(|counts| op.launches(&buffers, View::EMPTY, counts).unwrap())
                    .collect::<Vec<_>>();
                Ok((sparse_code(e), launches))
            },
        );
        // The decode attention: its heads, scale and paging, which its
        // query fixes, never the indexer's.
        let attention = s.attention(0, "attn_bf16_decode");
        let op = SparseAttentionOp {
            query: attention.query,
            max_context: s.max_context,
        };
        let buffers = SparseAttentionBuffers {
            q: view(1, BIG),
            cache: cache(false),
            block_tables: view(2, BIG),
            token_batch: view(3, BIG),
            positions: view(4, BIG),
            live_rows: view(5, 4),
            selected: view(6, BIG),
            workspace: view(7, BIG),
            output: view(8, BIG),
        };
        let mut classes = fixed.to_vec();
        classes.extend([
            ("block", Class::Selects),
            ("top_blocks", Class::Selects),
            ("max_context", Class::Selects),
            ("q_heads", Class::Selects),
            ("kv_heads", Class::Selects),
            ("head_dim", Class::Selects),
            ("page_size", Class::Selects),
            ("softmax_scale", Class::Selects),
        ]);
        keys_follow_launches(
            &attention,
            sparse_names(),
            sparse_fields(&classes),
            Choice::launch,
            |e| {
                let op = backend(e).sparse_attention(&op, Need::rows(8))?;
                let launches = at(8)
                    .map(|counts| op.launches(&buffers, View::EMPTY, counts).unwrap())
                    .collect::<Vec<_>>();
                Ok((sparse_code(e), launches))
            },
        );
    }

    /// The gated key: the query selects the op; the symbol, ABI and code
    /// are launch facts; the profile and claims are not.
    #[test]
    fn gated_launch_keys_hold_what_they_launch() {
        use super::select::{Class, field_names, keys_follow_launches};
        let s = &SERVED;
        let (_, q_heads, kv_heads) = s.ranks[0];
        let op = s.kv_write_op(q_heads, kv_heads, DType::Bf16);
        let entry = s
            .selection()
            .gated
            .into_iter()
            .find(|entry| gated_shape(entry.query) == gated_shape(op.query))
            .unwrap();
        let other = Profile::synthetic(ProfileNames {
            name: "gated-sparse-other",
            tag: "gs_other",
        });
        let gate = s.kv_write_op(q_heads, kv_heads, DType::Fp8E4M3).query;
        let names = field_names!(gated::Entry {
            profile,
            query,
            symbol,
            params,
            body,
            invariance
        });
        let fields: Vec<super::select::Field<gated::Entry>> = vec![
            (
                "profile",
                Class::Inert,
                Box::new(move |e| e.profile = other),
            ),
            ("query", Class::Selects, Box::new(move |e| e.query = gate)),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_gated_twin_v1".into()),
            ),
            ("params", Class::Launch, Box::new(|e| e.params.swap(0, 1))),
            ("body", Class::Launch, Box::new(|e| e.body.push_str("//"))),
            (
                "invariance",
                Class::Inert,
                Box::new(|e| e.invariance = Invariance::NONE),
            ),
        ];
        let buffers = gated_buffers(false);
        keys_follow_launches(&entry, names, fields, Choice::launch, |entry| {
            let prepared = Sm120::new()
                .with_gated_sparse(GatedSparseEntries {
                    gated: vec![entry.clone()],
                    ..GatedSparseEntries::default()
                })
                .gated_kv_write(&op, Need::rows(8))?;
            let launches = [1, 8].map(|rows| {
                prepared
                    .launches(
                        &buffers,
                        View::EMPTY,
                        Counts::rows(rows).with_items(rows, 4096),
                    )
                    .unwrap()
            });
            Ok((format!("{:?}", entry.entry_spec()), launches))
        });
    }

    /// The selected partials' key: the query holds their geometry, KV and
    /// phase; the symbol is a launch fact; the profile and the capacity,
    /// which the selection settles, are not.
    #[test]
    fn partials_launch_keys_hold_what_they_launch() {
        use super::select::{Class, field_names, keys_follow_launches};
        let s = &SERVED;
        let entry = s.selection().partials.remove(0);
        let other = Profile::synthetic(ProfileNames {
            name: "gated-sparse-other",
            tag: "gs_other",
        });
        let names = field_names!(selected_partials::Entry {
            profile,
            name,
            geometry,
            kv,
            phase,
            capacity
        });
        let fields: Vec<super::select::Field<selected_partials::Entry>> = vec![
            (
                "profile",
                Class::Inert,
                Box::new(move |e| e.profile = other),
            ),
            (
                "name",
                Class::Launch,
                Box::new(|e| e.name.push_str("_twin")),
            ),
            (
                "geometry",
                Class::Selects,
                Box::new(|e| e.geometry.top_blocks /= 2),
            ),
            (
                "kv",
                Class::Selects,
                Box::new(|e| {
                    e.kv = if e.kv == DType::Bf16 {
                        DType::Fp8E4M3
                    } else {
                        DType::Bf16
                    }
                }),
            ),
            (
                "phase",
                Class::Selects,
                Box::new(|e| {
                    e.phase = if e.phase == kernel_core::contract::AttentionPhase::Decode {
                        kernel_core::contract::AttentionPhase::Prefill
                    } else {
                        kernel_core::contract::AttentionPhase::Decode
                    }
                }),
            ),
            ("capacity", Class::Inert, Box::new(|e| e.capacity *= 2)),
        ];
        let op = SelectedPartialsOp {
            query: entry.query(),
            softmax_scale: 0.0625,
            max_context: s.max_context,
        };
        let buffers = SelectedPartialsBuffers {
            q: view(1, BIG),
            cache: cache(entry.kv == DType::Fp8E4M3),
            block_tables: view(2, BIG),
            token_batch: view(3, BIG),
            positions: view(4, BIG),
            live_rows: view(5, 4),
            selected: view(6, BIG),
            max: view(7, BIG),
            den: view(8, BIG),
            num: view(9, BIG),
        };
        let key = |entry: &selected_partials::Entry| {
            Partial {
                entry,
                guarantees: Invariance::NONE,
            }
            .launch()
        };
        keys_follow_launches(&entry, names, fields, key, |entry| {
            let prepared = Sm120::new()
                .with_gated_sparse(GatedSparseEntries {
                    partials: vec![entry.clone()],
                    ..GatedSparseEntries::default()
                })
                .selected_attention_partials(&op, 1, 2, Need::rows(1))?;
            let launches = prepared
                .launches(
                    &buffers,
                    View::EMPTY,
                    Counts::rows(1).with_items(1, s.max_context),
                )
                .unwrap();
            Ok((format!("{:?}", entry.entry_spec()), launches))
        });
    }

    /// The auditor's cases: block top-k registrations differing only in
    /// theta, the norm epsilon or the index heads, which no top-k reads,
    /// are one entry in every order.
    #[test]
    fn block_top_k_twins_in_unread_fields_are_one() {
        let top_k = SERVED.indexer("block_topk");
        let theta = sparse::Entry {
            geometry: sparse::Geometry {
                theta: 1e6,
                ..top_k.geometry
            },
            ..top_k.clone()
        };
        let eps = sparse::Entry {
            geometry: sparse::Geometry {
                eps: 1e-5,
                ..top_k.geometry
            },
            ..top_k.clone()
        };
        let heads = sparse::Entry {
            geometry: sparse::Geometry {
                index_heads: if top_k.geometry.index_heads == 8 {
                    4
                } else {
                    8
                },
                ..top_k.geometry
            },
            ..top_k.clone()
        };
        heads.contract().unwrap();
        for order in super::select::orders(&[top_k.clone(), theta, eps, heads]) {
            let prepared = Sm120::new()
                .with_gated_sparse(GatedSparseEntries {
                    sparse: order,
                    ..GatedSparseEntries::default()
                })
                .sparse_block_top_k(&top_k.query, Need::rows(8))
                .unwrap();
            assert_eq!(prepared.info().symbols[0].0, top_k.symbol());
        }
    }

    /// The auditor's case: index query registrations differing only in the
    /// ring their kernel never reads are one entry in every order.
    #[test]
    fn index_queries_differing_in_the_unread_ring_are_one() {
        let query = SERVED.indexer("index_query");
        let twin = sparse::Entry {
            geometry: sparse::Geometry {
                ring: query.geometry.ring * 2,
                ..query.geometry
            },
            ..query.clone()
        };
        twin.contract().unwrap();
        for order in super::select::orders(&[query.clone(), twin]) {
            let prepared = Sm120::new()
                .with_gated_sparse(GatedSparseEntries {
                    sparse: order,
                    ..GatedSparseEntries::default()
                })
                .sparse_index_query(&query.query, Need::rows(8))
                .unwrap();
            assert_eq!(prepared.info().symbols[0].0, query.symbol());
        }
    }

    /// The auditor's case: index query registrations differing only in the
    /// attention heads they never read are one entry in every order.
    #[test]
    fn index_queries_differing_in_unread_heads_are_one() {
        let query = SERVED.indexer("index_query");
        let twin = sparse::Entry {
            geometry: sparse::Geometry {
                q_heads: 12,
                ..query.geometry
            },
            ..query.clone()
        };
        twin.contract().unwrap();
        for order in super::select::orders(&[query.clone(), twin]) {
            let prepared = Sm120::new()
                .with_gated_sparse(GatedSparseEntries {
                    sparse: order,
                    ..GatedSparseEntries::default()
                })
                .sparse_index_query(&query.query, Need::rows(1))
                .unwrap();
            assert_eq!(prepared.info().symbols[0].0, query.symbol());
        }
    }

    #[test]
    fn sparse_attention_launches_what_the_builder_plans() {
        for s in SHAPES {
            let backend = s.backend();
            for rank in 0..2 {
                for (dtype, fp8) in [("bf16", false), ("fp8", true)] {
                    for (phase, capacity) in [("decode", 256), ("prefill", 8192)] {
                        let entry = s.attention(rank, &format!("attn_{dtype}_{phase}"));
                        let op = SparseAttentionOp {
                            query: entry.query,
                            max_context: s.max_context,
                        };
                        let prepared = backend.sparse_attention(&op, Need::rows(capacity)).unwrap();
                        let info = prepared.info();
                        assert_eq!(info.guarantees, entry.invariance);
                        // Prefill takes one split, so its merge never launches.
                        assert_eq!(info.symbols.len(), if phase == "prefill" { 1 } else { 2 });
                        assert_eq!(
                            info.bytes_of("workspace"),
                            (1..=capacity).map(|rows| entry.workspace_bytes(rows)).max()
                        );
                        let buffers = SparseAttentionBuffers {
                            q: view(1, BIG),
                            cache: cache(fp8),
                            block_tables: view(2, BIG),
                            token_batch: view(3, BIG),
                            positions: view(4, BIG),
                            live_rows: view(5, 4),
                            selected: view(6, BIG),
                            workspace: view(7, BIG),
                            output: view(8, BIG),
                        };
                        let [k_scale, v_scale] = scales(fp8);
                        for rows in (1..=64).chain([255, 256, capacity]) {
                            for context in [1, 2051, s.max_context] {
                                let splits = entry.splits(rows);
                                let mut legacy = vec![qsa(
                                    &entry,
                                    rows,
                                    context,
                                    vec![
                                        ptr(1),
                                        ptr(20),
                                        ptr(21),
                                        ptr(2),
                                        ptr(3),
                                        ptr(4),
                                        ptr(5),
                                        ptr(6),
                                        ptr(7),
                                        ptr(8),
                                        Argument::U32(s.max_pages()),
                                        Argument::U64(STRIDE),
                                        Argument::U64(STRIDE),
                                        Argument::U32(splits),
                                        Argument::U32(0.0625_f32.to_bits()),
                                        k_scale.clone(),
                                        v_scale.clone(),
                                    ],
                                )];
                                if let Some(plan) = entry.merge_plan(rows).unwrap() {
                                    legacy.push(launch(
                                        entry.merge_symbol().unwrap(),
                                        rows,
                                        plan,
                                        ArgumentList(vec![
                                            ptr(7),
                                            ptr(5),
                                            ptr(8),
                                            Argument::U32(splits),
                                        ]),
                                    ));
                                }
                                assert_eq!(
                                    prepared
                                        .launches(
                                            &buffers,
                                            View::EMPTY,
                                            Counts::rows(rows)
                                                .with_items(rows, context)
                                                .with_positions(context)
                                        )
                                        .unwrap(),
                                    legacy
                                );
                            }
                        }
                    }
                }
            }
            // Exact verify rows run the decode entry, which keeps one-row
            // bits up to the model's verify rows; the verify entry claims
            // none.
            let decode = s.attention(1, "attn_bf16_decode");
            let op = SparseAttentionOp {
                query: decode.query,
                max_context: s.max_context,
            };
            assert!(
                backend
                    .sparse_attention(&op, Need::rows(256).with_exact(one_row()))
                    .is_ok()
            );
            let verify = SparseAttentionOp {
                query: s.attention(1, "attn_bf16_verify").query,
                ..op
            };
            assert!(matches!(
                backend.sparse_attention(&verify, Need::rows(256).with_exact(one_row())),
                Err(Refused::Inexact { .. })
            ));
        }
    }

    /// A partials entry guarantees what the model declares for its symbol,
    /// and an undeclared one nothing; claims leave the launches as they are.
    #[test]
    fn selected_partials_report_declared_claims() {
        let claim = Invariance {
            rows: Rows::OneRow { max: 4 },
            ..Invariance::NONE
        };
        for s in SHAPES {
            let partials = s.selection().partials;
            let verify = partials
                .iter()
                .find(|entry| entry.phase == kernel_core::contract::AttentionPhase::Verify)
                .unwrap();
            let decode = partials
                .iter()
                .find(|entry| {
                    entry.phase == kernel_core::contract::AttentionPhase::Decode
                        && entry.kv == verify.kv
                })
                .unwrap();
            let backend = s
                .backend()
                .with_context_parallel_claims([(verify.symbol(), claim)]);
            let op = |entry: &selected_partials::Entry| SelectedPartialsOp {
                query: entry.query(),
                softmax_scale: 0.0625,
                max_context: s.max_context,
            };
            let claimed = backend
                .selected_attention_partials(&op(verify), 1, 2, Need::rows(4).with_exact(claim))
                .unwrap();
            assert_eq!(claimed.info().guarantees, claim);
            let plain = s
                .backend()
                .selected_attention_partials(&op(verify), 1, 2, Need::rows(4))
                .unwrap();
            assert_eq!(plain.info().implementation, claimed.info().implementation);
            assert!(matches!(
                backend.selected_attention_partials(
                    &op(decode),
                    1,
                    2,
                    Need::rows(4).with_exact(claim)
                ),
                Err(Refused::Inexact { .. })
            ));
        }
    }

    #[test]
    fn selected_partials_launch_what_the_builder_plans() {
        for s in SHAPES {
            let backend = s.backend();
            for entry in s.selection().partials {
                let op = SelectedPartialsOp {
                    query: entry.query(),
                    softmax_scale: 0.0625,
                    max_context: s.max_context,
                };
                let fp8 = entry.kv == DType::Fp8E4M3;
                let prepared = backend
                    .selected_attention_partials(&op, 1, 2, Need::rows(entry.capacity))
                    .unwrap();
                assert_eq!(prepared.info().guarantees, Invariance::NONE);
                let buffers = SelectedPartialsBuffers {
                    q: view(1, BIG),
                    cache: cache(fp8),
                    block_tables: view(2, BIG),
                    token_batch: view(3, BIG),
                    positions: view(4, BIG),
                    live_rows: view(5, 4),
                    selected: view(6, BIG),
                    max: view(7, BIG),
                    den: view(8, BIG),
                    num: view(9, BIG),
                };
                let [k_scale, v_scale] = scales(fp8);
                for rows in [1, 3, entry.capacity] {
                    assert_eq!(
                        prepared
                            .launches(
                                &buffers,
                                View::EMPTY,
                                Counts::rows(rows)
                                    .with_items(1, s.max_context)
                                    .with_positions(s.max_context)
                            )
                            .unwrap(),
                        [launch(
                            entry.symbol(),
                            rows,
                            entry.plan(rows).unwrap(),
                            ArgumentList(vec![
                                ptr(1),
                                ptr(20),
                                ptr(21),
                                ptr(2),
                                ptr(3),
                                ptr(4),
                                ptr(5),
                                ptr(6),
                                ptr(7),
                                ptr(8),
                                ptr(9),
                                Argument::U32(s.max_pages()),
                                Argument::U64(STRIDE),
                                Argument::U64(STRIDE),
                                Argument::U32(0.0625_f32.to_bits()),
                                k_scale.clone(),
                                v_scale.clone(),
                                Argument::U32(1),
                                Argument::U32(2),
                            ])
                        )]
                    );
                }
                assert!(
                    backend
                        .selected_attention_partials(&op, 2, 2, Need::rows(1))
                        .is_err()
                );
            }
        }
    }

    #[test]
    fn prepare_refuses_unserved_over_capacity_and_inexact_ops() {
        let s = &SERVED;
        let backend = s.backend();
        let op = s.kv_write_op(6, 1, DType::Bf16);
        // No selection, and a shape no entry instantiates.
        assert!(matches!(
            Sm120::new().gated_kv_write(&op, Need::rows(1)),
            Err(Refused::Rule(rule)) if rule.contains("no SM120")
        ));
        let gate = OutputGateOp {
            query: GatedAttentionQuery::OutputGate {
                heads: 6,
                head_dim: 256,
                capacity: 1,
            },
            kv_heads: 2,
        };
        assert!(matches!(
            backend.output_gate(&gate, Need::rows(1)),
            Err(Refused::Rule(rule)) if rule.contains("no SM120")
        ));
        // Past every entry's rows and past what SM120 prepares.
        for capacity in [8193, u32::MAX, 0] {
            assert!(matches!(
                backend.gated_kv_write(&op, Need::rows(capacity)),
                Err(Refused::Rule(_))
            ));
        }
        let attention = SparseAttentionOp {
            query: s.attention(0, "attn_bf16_decode").query,
            max_context: s.max_context,
        };
        assert!(matches!(
            backend.sparse_attention(&attention, Need::rows(257)),
            Err(Refused::Rule(rule)) if rule.contains("holds 257 rows")
        ));
        // A different context changes the block table rows the entry reads.
        assert!(matches!(
            backend.sparse_attention(
                &SparseAttentionOp {
                    max_context: 8192,
                    ..attention
                },
                Need::rows(1)
            ),
            Err(Refused::Rule(_))
        ));
        // Exactness no entry keeps: gated rows claim no tie rule, prefill
        // attention no one-row bits.
        assert!(matches!(
            backend.gated_kv_write(
                &op,
                Need::rows(64).with_exact(Invariance::NONE.with_ties(Ties::LowerIndex))
            ),
            Err(Refused::Inexact { .. })
        ));
        let prefill = SparseAttentionOp {
            query: s.attention(0, "attn_bf16_prefill").query,
            ..attention
        };
        assert!(matches!(
            backend.sparse_attention(&prefill, Need::rows(64).with_exact(Invariance::one_row(4))),
            Err(Refused::Inexact { .. })
        ));
        // Head widths the SM120 templates lack: math to the contract,
        // refused at prepare, and still refused by the builders.
        let wide = gated::Config {
            head_dim: 192,
            ..s.config(6, 1)
        };
        let wide_entries = gated::entries(s.ranks[0].0, wide);
        assert!(wide_entries.iter().all(|entry| entry.contract().is_err()));
        let wide_op = GatedKvWriteOp {
            query: wide_entries[0].query,
            max_context: s.max_context,
        };
        assert_eq!(wide_op.validate(), Ok(()));
        assert!(matches!(
            backend.gated_kv_write(&wide_op, Need::rows(1)),
            Err(Refused::Rule(rule)) if rule.contains("head dim 128 or 256")
        ));
        let wide_geometry = sparse::Geometry {
            head_dim: 192,
            ..s.geometry(6)
        };
        let wide_attention = sparse::attention_entries(s.ranks[0].0, wide_geometry, EXACT_ROWS);
        assert!(wide_attention.iter().all(|entry| entry.contract().is_err()));
        let wide_op = SparseAttentionOp {
            query: wide_attention[0].query,
            max_context: s.max_context,
        };
        assert_eq!(wide_op.validate(), Ok(()));
        assert!(matches!(
            backend.sparse_attention(&wide_op, Need::rows(1)),
            Err(Refused::Rule(rule)) if rule.contains("outside the SM120 engine")
        ));
        // Ops of the wrong query kind.
        assert!(matches!(
            backend.sparse_index_query(&attention.query, Need::rows(1)),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            backend.sparse_block_top_k(&attention.query, Need::rows(1)),
            Err(Refused::Rule(_))
        ));
    }

    #[test]
    fn launches_check_their_buffers_and_counts() {
        let s = &SERVED;
        let backend = s.backend();
        let op = s.kv_write_op(6, 1, DType::Fp8E4M3);
        let prepared = backend.gated_kv_write(&op, Need::rows(64)).unwrap();
        let counts = Counts::rows(64).with_items(64, 4096).with_positions(4096);
        assert!(
            prepared
                .launches(&gated_buffers(true), View::EMPTY, counts)
                .is_ok()
        );
        let short = |mutate: &dyn Fn(&mut GatedKvWriteBuffers<View>)| {
            let mut buffers = gated_buffers(true);
            mutate(&mut buffers);
            prepared.launches(&buffers, View::EMPTY, counts).is_err()
        };
        // Packed rows of 6 Q heads (query and gate) and one K and V head.
        let qkv = 64 * (2 * 6 + 2) * 256 * 2;
        assert!(!short(&|b| b.qkv.bytes = qkv));
        assert!(short(&|b| b.qkv.bytes = qkv - 1));
        assert!(short(&|b| b.q.bytes = 64 * 6 * 256 * 2 - 1));
        assert!(short(&|b| b.rope.cos_sin = None));
        assert!(short(&|b| b.rope.cos_sin = Some(view(7, 4096 * 64 * 4 - 1))));
        assert!(short(&|b| b.block_tables.bytes = 64 * 4096 * 4 - 1));
        assert!(short(&|b| b.cache.k_slot_stride = 64 * 256 - 1));
        assert!(short(&|b| b.cache.v.bytes = 63 * STRIDE));
        assert!(short(&|b| b.cache.k_scale = 0.0));
        assert!(
            prepared
                .launches(&gated_buffers(true), View::EMPTY, Counts::rows(65))
                .is_err()
        );
        assert!(
            prepared
                .launches(&gated_buffers(true), View::EMPTY, Counts::rows(0))
                .is_err()
        );
        // The rotary table covers the step's M-RoPE positions, not its
        // context.
        let mut table = gated_buffers(true);
        table.rope.cos_sin = Some(view(7, 2048 * 64 * 4));
        assert!(prepared.launches(&table, View::EMPTY, counts).is_err());
        assert!(
            prepared
                .launches(&table, View::EMPTY, counts.with_positions(2048))
                .is_ok()
        );

        // Sparse ops refuse a context past the prepared one and short
        // workspaces.
        let entry = s.attention(0, "attn_bf16_decode");
        let prepared = backend
            .sparse_attention(
                &SparseAttentionOp {
                    query: entry.query,
                    max_context: s.max_context,
                },
                Need::rows(8),
            )
            .unwrap();
        let buffers = SparseAttentionBuffers {
            q: view(1, BIG),
            cache: cache(false),
            block_tables: view(2, BIG),
            token_batch: view(3, BIG),
            positions: view(4, BIG),
            live_rows: view(5, 4),
            selected: view(6, BIG),
            workspace: view(7, entry.workspace_bytes(8)),
            output: view(8, BIG),
        };
        let full = Counts::rows(8)
            .with_items(8, s.max_context)
            .with_positions(s.max_context);
        assert!(prepared.launches(&buffers, View::EMPTY, full).is_ok());
        for counts in [
            Counts::rows(8)
                .with_items(8, s.max_context + 1)
                .with_positions(s.max_context + 1),
            Counts::rows(8).with_items(8, 0).with_positions(0),
            Counts::rows(9).with_items(8, 100).with_positions(100),
        ] {
            assert!(prepared.launches(&buffers, View::EMPTY, counts).is_err());
        }
        let short = SparseAttentionBuffers {
            workspace: view(7, entry.workspace_bytes(8) - 1),
            ..buffers
        };
        assert!(prepared.launches(&short, View::EMPTY, full).is_err());
        let selected = SparseAttentionBuffers {
            selected: view(6, 8 * 512 * 4 - 1),
            ..buffers
        };
        assert!(prepared.launches(&selected, View::EMPTY, full).is_err());
    }
}
