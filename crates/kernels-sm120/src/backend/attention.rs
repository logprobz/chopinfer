//! Dense attention through the SM120 attention variants: paged and ring KV
//! in prefill, decode and verify, exact verify rows and folded RoPE.

use kernel_core::attention::{AttentionBuffers, AttentionFold, AttentionOp};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{
    AttentionPhase, AttentionQuery, DType, Refusal, RotarySource, RuntimeCounts, Validate,
};
use kernel_core::properties::{Invariance, Rows};

use super::check_plan;
use super::select::{Candidate, Chosen, choose};
use crate::attention::{
    Buffers, CacheLayout, Choice, EXACT_ROWS, Fold, GatedFold, GatedRopeInputs, MAX_VERIFY_TOKENS,
    PlannedAttention, RopeInputs, Variant,
};

/// The attention tuning table, whose split limits, CTA counts and merge
/// widths are numerics choices of every op.
const TUNING: &str = include_str!("../../tuning/attention.json");

/// A model's SM120 attention variants and its choices among paths that
/// give the same bits.
#[derive(Clone, Debug, Default)]
pub struct AttentionEntries {
    /// The variants the model's cubins instantiate. Of the variants that
    /// serve an op and declare the exactness it needs, it runs the one with
    /// the least query capacity holding its item rows, then the least batch
    /// capacity holding its items (a variant of fewer items is prepared for
    /// the op's, after every variant that holds them); two such variants
    /// that would launch differently are a tie, and the op is refused.
    pub variants: Vec<Variant>,
    /// Exact verify rows of paged attention run the per-row entries, not
    /// the shared-key row-group entries: the same bits at more cost on long
    /// contexts.
    pub per_row_exact: bool,
    /// Verify steps of more than this many rows that a decode variant's
    /// exact rows entries serve run those entries per row, whatever
    /// exactness they need (a diagnostic). Other verify ops, such as
    /// bidirectional block attention, keep their verify entries.
    pub rows_above: Option<u32>,
}

/// The attention `op` through the variant of `entries` that serves it. A
/// verify step that needs one-row bits runs the matching decode variant's
/// exact rows entries.
pub(super) fn prepare(
    entries: Option<&AttentionEntries>,
    op: &AttentionOp,
    need: Need,
) -> Result<Prepared<AttentionBuffers<View>>, Refused> {
    let entries = entries.ok_or(Refused::Unsupported)?;
    op.validate()?;
    crate::attention::check_query(&op.query)?;
    let verify = op.query.phase == AttentionPhase::Verify;
    let exact = verify && need.exact.rows != Rows::None;
    if exact && op.fold.is_some() {
        return Err(Refused::Rule("exact verify rows fold no RoPE".into()));
    }
    let wide = verify
        && op.fold.is_none()
        && entries
            .rows_above
            .is_some_and(|above| need.capacity > above);
    // A verify step that needs one-row bits runs the decode rows entries,
    // as wide steps do under the diagnostic where they exist.
    let decode = AttentionQuery {
        phase: AttentionPhase::Decode,
        ..op.query
    };
    let items = op.query.max_batch_capacity;
    // The rows path of a variant: shared-key row groups for exact rows of
    // a paged cache unless the model runs each row alone.
    let rows_path = |v: &Variant| {
        if exact && v.layout == CacheLayout::Paged && !entries.per_row_exact {
            Path::RowsShared
        } else {
            Path::Rows
        }
    };
    let rows = (exact || wide)
        .then(|| {
            let served = entries
                .variants
                .iter()
                .filter(|v| v.has_rows() && serves(v, &decode, 1))
                .map(|v| (v, rows_path(v)))
                .collect::<Vec<_>>();
            (!served.is_empty()).then(|| chosen(served, 1, items, need.exact))
        })
        .flatten()
        .transpose()?;
    let Chosen {
        entry: (variant, path),
        guarantees,
    } = match rows {
        Some(chosen) => chosen,
        None if exact => return Err(no_entry()),
        None => {
            let path = Path::of(op.fold);
            let served = entries
                .variants
                .iter()
                .filter(|v| serves(v, &op.query, op.item_rows) && path.folded_by(v))
                .map(|v| (v, path))
                .collect();
            chosen(served, op.item_rows, items, need.exact)?
        }
    };
    let mut op = AttentionPrepared::new(variant, path, *op, need.capacity)?;
    // The op claims no more than every registration of its launch declares.
    op.info.guarantees = op.info.guarantees.meet(guarantees);
    need.check(&op.info)?;
    Ok(Box::new(op))
}

fn no_entry() -> Refused {
    Refused::Rule("no SM120 attention entry serves the op".into())
}

/// What every launch of `symbols` keeps: the meet of their declarations in
/// `declared`. A symbol declared more than once is one launch, claiming the
/// meet of its declarations, as the select rule treats entries of equal
/// launches. Refuses a symbol `declared` does not hold.
fn launched_guarantees(
    declared: &[(String, Invariance)],
    symbols: &[String],
) -> Result<Invariance, Refused> {
    let mut guarantees = None;
    for symbol in symbols {
        let invariance = declared
            .iter()
            .filter(|(declared, _)| declared == symbol)
            .map(|&(_, invariance)| invariance)
            .reduce(Invariance::meet)
            .ok_or_else(|| Refused::Rule("an attention entry is not declared".into()))?;
        guarantees = Some(guarantees.map_or(invariance, |held: Invariance| held.meet(invariance)));
    }
    Ok(guarantees.unwrap_or(Invariance::NONE))
}

/// The variant of `served` (each with the path it would run) for items of
/// up to `item_rows` rows, `items` of them, needing `exact`: of the
/// variants whose path declares `exact`, the least query capacity that
/// holds the rows, then the least batch capacity that holds the items. A
/// variant of fewer items ranks after those, prepared for the op's items
/// with the same entries. Refuses a tie.
fn chosen(
    served: Vec<(&Variant, Path)>,
    item_rows: u32,
    items: u32,
    exact: Invariance,
) -> Result<Chosen<(Variant, Path)>, Refused> {
    let mut unextended = None;
    let candidates = served
        .into_iter()
        .filter_map(|(v, path)| {
            let holds = items <= v.query.max_batch_capacity;
            let variant = if holds {
                *v
            } else {
                match v.with_batch_capacity(items) {
                    Ok(variant) => variant,
                    Err(refusal) => {
                        unextended.get_or_insert(refusal);
                        return None;
                    }
                }
            };
            let rank = if holds {
                u64::from(v.query.max_batch_capacity)
            } else {
                u64::MAX
            };
            let candidate = Candidate::new(
                (variant, path),
                v.name,
                launch(&variant, path),
                declared(&variant, path),
            );
            Some(candidate.capacity(v.query_capacity).rank(rank))
        })
        .collect::<Vec<_>>();
    if let (true, Some(refusal)) = (candidates.is_empty(), unextended) {
        return Err(refusal.into());
    }
    choose(
        "attention",
        candidates,
        Need {
            capacity: item_rows,
            exact,
        },
    )
}

/// What a variant launches on `path`: the entries its name gives on that
/// path, its query (with the batch capacity its plans are prepared for),
/// cache layout and row capacity. The path states its own fold, and a
/// variant serves it only with that fold's entry (`Path::folded_by`) or,
/// for the rows paths, its exact rows entries (`Variant::has_rows`); the
/// folds and exact rows entries a path does not launch are not its facts.
fn launch(variant: &Variant, path: Path) -> String {
    let Variant {
        name,           // the entries' symbols
        profile: _,     // filters catalogs; no launch reads it
        query,          // the plans; its shape selects the op
        layout,         // the cache the entries read
        query_capacity, // the plans
        rope_fold,      // the NeoX fold entry, on that path only
        gated_fold,     // the gated fold entry, on that path only
        invariance: _,  // claims
        exact_rows,     // the exact rows entries, on those paths only
    } = variant;
    let launched = match path {
        Path::Phase => String::new(),
        Path::Neox { .. } => format!("{rope_fold:?}"),
        Path::Gated(_) => format!("{gated_fold:?}"),
        Path::Rows | Path::RowsShared => format!("{exact_rows}"),
    };
    format!("{name} {path:?} {query:?} {layout:?} {query_capacity} {launched}")
}

/// What `path`'s entries of `variant` declare together, as the prepared op
/// meets their catalog declarations (`Variant::declared_symbols`): the
/// phase entries' declaration, one-row bits for the exact rows entries,
/// and nothing for the entries that fold RoPE in.
fn declared(variant: &Variant, path: Path) -> Invariance {
    match path {
        Path::Phase => variant.invariance,
        Path::Rows | Path::RowsShared => Invariance {
            rows: Rows::OneRow { max: EXACT_ROWS },
            ..Invariance::NONE
        },
        Path::Neox { .. } | Path::Gated(_) => Invariance::NONE,
    }
}

/// Whether `variant` attends as `query` over items of up to `item_rows`
/// query rows, at the query's items or more.
fn serves(variant: &Variant, query: &AttentionQuery, item_rows: u32) -> bool {
    variant.query_capacity >= item_rows
        && AttentionQuery {
            max_batch_capacity: variant.query.max_batch_capacity,
            ..*query
        } == variant.query
}

/// The entries a prepared op launches.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    /// The variant's phase entries: a partial (or direct) entry, then a
    /// merge unless the plan writes its output directly.
    Phase,
    /// The decode entry that folds NeoX RoPE and the KV write in, with
    /// computed angles or (`table`) a cos/sin table.
    Neox {
        rotary: u32,
        theta: f32,
        v_multiplier: f32,
        table: bool,
    },
    /// The decode entry that folds the gated norm, M-RoPE and the KV write
    /// in.
    Gated(GatedFold),
    /// Each verify row as the lone row of a decode step.
    Rows,
    /// As `Rows`, with up to four rows of an item per CTA over one read of
    /// their keys.
    RowsShared,
}

impl Path {
    fn of(fold: Option<AttentionFold>) -> Path {
        match fold {
            None => Path::Phase,
            Some(AttentionFold::Neox {
                rotary,
                theta,
                v_multiplier,
                source,
            }) => Path::Neox {
                rotary,
                theta,
                v_multiplier,
                table: source == RotarySource::CosSinTable,
            },
            Some(AttentionFold::Gated {
                rotary,
                sections,
                eps,
            }) => Path::Gated(GatedFold {
                rotary,
                sections,
                eps,
            }),
        }
    }

    /// Whether `variant` has this path's fold entry.
    fn folded_by(self, variant: &Variant) -> bool {
        match self {
            Path::Neox { rotary, .. } => variant.rope_fold == Some(rotary),
            Path::Gated(fold) => variant.gated_fold == Some(fold),
            Path::Phase | Path::Rows | Path::RowsShared => true,
        }
    }

    fn rows(self) -> bool {
        matches!(self, Path::Rows | Path::RowsShared)
    }

    /// The rotary width whose cos/sin table the path reads, if any.
    fn table(self) -> Option<u32> {
        match self {
            Path::Neox {
                rotary,
                table: true,
                ..
            } => Some(rotary),
            Path::Gated(fold) => Some(fold.rotary),
            _ => None,
        }
    }
}

/// An attention through one SM120 variant, whose bucket choices (splits,
/// CTA targets, piece units, plans) are made from the tuning table when it
/// is prepared. A launch binds the step's buffers to its bucket's choice:
/// decode and verify at the smallest power of two that holds the step's
/// items, prefill at the prepared items (with one split for short steps),
/// exact rows at one item spread over the step's rows.
struct AttentionPrepared {
    variant: Variant,
    path: Path,
    /// The most query rows of a launch.
    capacity: u32,
    /// The most items of a step: the op's, which may be fewer than the
    /// variant's bucket holds.
    items: u32,
    /// Whether every step plans for `items`, whatever items it serves
    /// (`AttentionOp::plan_at_capacity`).
    at_capacity: bool,
    /// Decode and verify: one choice per bucket, by its power of two.
    /// Prefill: the long and the short choice. Exact rows: bucket one's.
    choices: Vec<Choice>,
    info: OpInfo,
}

impl AttentionPrepared {
    /// Makes every choice the op can launch, then plans each at full counts
    /// over placeholder buffers to list its entries and size its workspace.
    fn new(variant: Variant, path: Path, op: AttentionOp, capacity: u32) -> Result<Self, Refused> {
        variant.validate()?;
        let query = variant.query;
        let items = op.query.max_batch_capacity;
        // Steps pad their rows to the bucket that holds their items; the
        // item limit itself applies at launch.
        let padded = items.next_power_of_two();
        let limit = match query.phase {
            AttentionPhase::Decode if !path.rows() => padded,
            AttentionPhase::Prefill => variant.query_capacity,
            // Verify, on its own entries or the decode rows entries.
            _ => padded.saturating_mul(op.item_rows).min(MAX_VERIFY_TOKENS),
        };
        if capacity == 0 || capacity > limit {
            return Err(Refused::Rule(
                "attention rows exceed what the SM120 entry holds".into(),
            ));
        }
        let choices = if path.rows() {
            vec![variant.choose(1, false)?]
        } else if query.phase == AttentionPhase::Prefill {
            let bucket = query.max_batch_capacity;
            vec![
                variant.choose(bucket, false)?,
                variant.choose(bucket, true)?,
            ]
        } else {
            let buckets = 0..=items.next_power_of_two().trailing_zeros();
            buckets
                .map(|shift| variant.choose(1 << shift, false))
                .collect::<Result<_, _>>()?
        };
        let mut op = Self {
            variant,
            path,
            capacity,
            items,
            at_capacity: op.plan_at_capacity,
            choices,
            info: OpInfo::new("", Invariance::NONE, Vec::new()),
        };
        let buffers = placeholder(variant)?;
        let mut planned = Vec::new();
        for choice in &op.choices {
            let live = match query.phase {
                _ if path.rows() => live(query.max_context_capacity, 1, 1),
                AttentionPhase::Prefill if choice.splits == 1 => live(1, 1, choice.prepared_batch),
                _ => live(query.max_context_capacity, 1, choice.prepared_batch),
            };
            planned.push(op.plan(choice, capacity, live, buffers.clone(), FOLD)?);
        }
        let mut symbols: Vec<String> = Vec::new();
        let mut workspace = 0;
        for planned in &planned {
            check_plan(&planned.partial)?;
            workspace = workspace.max(planned.partial.workspace_bytes);
            let mut launched = vec![&planned.partial_symbol];
            if !planned.direct_output {
                check_plan(&planned.merge)?;
                launched.push(&planned.merge_symbol);
            }
            for symbol in launched {
                if !symbols.contains(symbol) {
                    symbols.push(symbol.clone());
                }
            }
        }
        // Partials before merges, the order a step launches them.
        symbols.sort_by_key(|symbol| planned.iter().any(|p| &p.merge_symbol == symbol));
        let mut shared = Vec::with_capacity(symbols.len());
        for symbol in &symbols {
            let bytes = variant
                .dynamic_shared_bytes(symbol)
                .ok_or_else(|| Refused::Rule("an attention entry is not the variant's".into()))?;
            shared.push((symbol.clone(), bytes));
        }
        let guarantees = launched_guarantees(&variant.declared_symbols(), &symbols)?;
        let digest = kernel_build::digest(format!("{:?}{TUNING}", (variant, path)).as_bytes());
        op.info = OpInfo::new(
            format!(
                "sm120.attention.v1[{}]#{}",
                symbols.join(","),
                &digest[..16]
            ),
            guarantees,
            shared,
        );
        op.info.bytes = vec![("workspace".to_owned(), workspace)];
        Ok(op)
    }

    /// The choice a step of `items` items and `live` counts launches with.
    fn choice(&self, items: u32, live: RuntimeCounts) -> &Choice {
        let index = if self.path.rows() {
            0
        } else if self.variant.query.phase == AttentionPhase::Prefill {
            usize::from(self.variant.short_prefill(live))
        } else {
            items.next_power_of_two().trailing_zeros() as usize
        };
        &self.choices[index]
    }

    /// The launch of `choice` over `buffers`, as the builder plans it.
    fn plan(
        &self,
        choice: &Choice,
        rows: u32,
        live: RuntimeCounts,
        buffers: Buffers,
        fold: FoldViews,
    ) -> Result<PlannedAttention, Refusal> {
        let v = self.variant;
        let qkv = buffers.q;
        let rope = |theta, v_multiplier| {
            Some(Fold::Neox(RopeInputs {
                qkv,
                theta,
                v_multiplier,
            }))
        };
        match self.path {
            Path::Phase => v.plan_chosen(choice, live, buffers, None),
            Path::Neox {
                theta,
                v_multiplier,
                table,
                ..
            } => {
                let planned = v.plan_chosen(choice, live, buffers, rope(theta, v_multiplier))?;
                Ok(if table {
                    v.with_table(planned, fold.cos_sin)
                } else {
                    planned
                })
            }
            Path::Gated(_) => {
                let rope = GatedRopeInputs {
                    qkv,
                    q_norm: fold.q_norm,
                    k_norm: fold.k_norm,
                    rope_positions: fold.rope_positions,
                    cos_sin: fold.cos_sin,
                };
                v.plan_chosen(choice, live, buffers, Some(Fold::Gated(rope)))
            }
            Path::Rows | Path::RowsShared => {
                if self.path == Path::RowsShared {
                    v.check_rows_shared(rows)?;
                }
                v.check_rows(rows, &buffers)?;
                let one = RuntimeCounts { batch: 1, ..live };
                let planned = v.spread_rows(v.plan_chosen(choice, one, buffers, None)?, rows)?;
                if self.path == Path::RowsShared {
                    v.share_rows(planned, rows)
                } else {
                    Ok(planned)
                }
            }
        }
    }
}

/// A fold's input handles.
#[derive(Clone, Copy)]
struct FoldViews {
    cos_sin: u64,
    q_norm: u64,
    k_norm: u64,
    rope_positions: u64,
}

/// Placeholder fold inputs for planning up front.
const FOLD: FoldViews = FoldViews {
    cos_sin: 9,
    q_norm: 10,
    k_norm: 11,
    rope_positions: 12,
};

fn live(context: u32, m: u32, batch: u32) -> RuntimeCounts {
    RuntimeCounts {
        m,
        batch,
        context,
        routed_rows: 0,
        split_k: 1,
    }
}

fn element_bytes(query: &AttentionQuery) -> u64 {
    if query.kv_dtype == DType::Bf16 { 2 } else { 1 }
}

/// Bytes of one page or ring of `dim`-wide heads.
fn slot_bytes(query: &AttentionQuery, dim: u32) -> u64 {
    u64::from(query.page_size) * u64::from(query.kv_heads) * u64::from(dim) * element_bytes(query)
}

/// Buffers that every phase accepts, over one tightly packed slot.
fn placeholder(variant: Variant) -> Result<Buffers, Refusal> {
    let (k_map, v_map) = variant.cache_maps(1, 1, 2)?;
    Ok(Buffers {
        q: 1,
        k_cache: 1,
        v_cache: 2,
        k_page_stride_bytes: slot_bytes(&variant.query, variant.query.qk_dim),
        v_page_stride_bytes: slot_bytes(&variant.query, variant.query.v_dim),
        k_map: Some(k_map),
        v_map: Some(v_map),
        block_tables: Some(3),
        slot_ids: Some(3),
        seq_lens: Some(4),
        q_start: Some(4),
        q_len: Some(4),
        prefix_len: Some(4),
        live_batch: 5,
        k_new: Some(6),
        v_new: Some(6),
        sink: Some(7),
        output: 8,
        workspace: 8,
        k_scale: 1.0,
        v_scale: 1.0,
    })
}

impl PreparedOp<AttentionBuffers<View>> for AttentionPrepared {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &AttentionBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let query = self.variant.query;
        let (rows, items) = (counts.rows, counts.items);
        // A rows path spreads its grid over the bucket a step pads its rows
        // to; a row past the device live rows writes nothing.
        let grid_rows = if self.path.rows() {
            rows.max(counts.padded_rows)
        } else {
            rows
        };
        if rows == 0 || grid_rows > self.capacity {
            return Err(Refusal("attention rows exceed the prepared capacity"));
        }
        if items == 0 || items > self.items {
            return Err(Refusal("attention items exceed the prepared items"));
        }
        // A decode row is an item's query row.
        if query.phase == AttentionPhase::Decode && !self.path.rows() && items > rows {
            return Err(Refusal("decode items exceed the step's rows"));
        }
        let q_rows = u64::from(rows);
        let (q_heads, kv_heads) = (u64::from(query.q_heads), u64::from(query.kv_heads));
        let (qk, v) = (u64::from(query.qk_dim), u64::from(query.v_dim));
        let q_width = match self.path {
            Path::Neox { .. } => q_heads * qk + kv_heads * (qk + v),
            Path::Gated(_) => 2 * q_heads * qk + kv_heads * (qk + v),
            _ => q_heads * qk,
        };
        buffers.q.covers(q_rows * q_width * 2)?;
        buffers.output.covers(q_rows * q_heads * v * 2)?;
        let cache = buffers.cache;
        if cache.slots == 0 {
            return Err(Refusal("a KV cache holds at least one slot"));
        }
        // Ring kernels address a slot's ring as tightly packed after the
        // last, whatever strides the arguments carry.
        let (k_slot, v_slot) = (
            slot_bytes(&query, query.qk_dim),
            slot_bytes(&query, query.v_dim),
        );
        if self.variant.layout == CacheLayout::Ring
            && (cache.k_slot_stride != k_slot || cache.v_slot_stride != v_slot)
        {
            return Err(Refusal("ring KV slots are tightly packed"));
        }
        for (plane, stride, bytes) in [
            (cache.k, cache.k_slot_stride, k_slot),
            (cache.v, cache.v_slot_stride, v_slot),
        ] {
            let bytes = u64::from(cache.slots - 1)
                .checked_mul(stride)
                .and_then(|offset| offset.checked_add(bytes))
                .ok_or(Refusal("KV cache bytes overflow u64"))?;
            plane.covers(bytes)?;
        }
        let pages = u64::from(query.max_context_capacity.div_ceil(query.page_size));
        if let Some(tables) = buffers.block_tables {
            tables.covers(u64::from(items) * pages * 4)?;
        }
        let per_item = [
            buffers.slot_ids,
            buffers.seq_lens,
            buffers.q_start,
            buffers.q_len,
            buffers.prefix_len,
        ];
        for view in per_item.into_iter().flatten() {
            view.covers(u64::from(items) * 4)?;
        }
        buffers.live_batch.covers(4)?;
        for (view, dim) in [(buffers.k_new, qk), (buffers.v_new, v)] {
            if let Some(view) = view {
                view.covers(u64::from(rows) * kv_heads * dim * 2)?;
            }
        }
        if let Some(sink) = buffers.sink {
            sink.covers(q_heads * 2)?;
        }
        let rope = buffers.rope;
        let mut fold = FOLD;
        if let Some(rotary) = self.path.table() {
            let cos_sin = rope
                .and_then(|rope| rope.cos_sin)
                .ok_or(Refusal("folded RoPE table missing"))?;
            // Every rotary position of the step: rotary / 2 pairs of F32
            // cos and sin each.
            cos_sin.covers(u64::from(counts.positions.max(1)) * u64::from(rotary) * 4)?;
            fold.cos_sin = cos_sin.handle;
        }
        if let Path::Gated(_) = self.path {
            let (Some(q_norm), Some(k_norm), Some(positions)) = (
                rope.and_then(|rope| rope.q_norm),
                rope.and_then(|rope| rope.k_norm),
                rope.and_then(|rope| rope.rope_positions),
            ) else {
                return Err(Refusal("gated fold inputs missing"));
            };
            q_norm.covers(qk * 2)?;
            k_norm.covers(qk * 2)?;
            positions.covers(q_rows * 3 * 4)?;
            fold.q_norm = q_norm.handle;
            fold.k_norm = k_norm.handle;
            fold.rope_positions = positions.handle;
        }
        // The per-item arrays hold the step's items; an op planned at its
        // capacity launches as for all of its items.
        let planned_items = if self.at_capacity { self.items } else { items };
        let live = RuntimeCounts {
            m: rows,
            batch: planned_items,
            context: counts.context,
            routed_rows: 0,
            split_k: 1,
        };
        let choice = self.choice(planned_items, live);
        let handle = |view: Option<View>| view.map(|view| view.handle);
        // Only prefill reads the cache through tensor maps.
        let maps = if query.phase == AttentionPhase::Prefill {
            let (k, v) = self.variant.maps(
                choice.key_tile,
                cache.slots,
                cache.k.handle,
                cache.v.handle,
                cache.k_slot_stride,
                cache.v_slot_stride,
            )?;
            (Some(k), Some(v))
        } else {
            (None, None)
        };
        let legacy = Buffers {
            q: buffers.q.handle,
            k_cache: cache.k.handle,
            v_cache: cache.v.handle,
            k_page_stride_bytes: cache.k_slot_stride,
            v_page_stride_bytes: cache.v_slot_stride,
            k_map: maps.0,
            v_map: maps.1,
            block_tables: handle(buffers.block_tables),
            slot_ids: handle(buffers.slot_ids),
            seq_lens: handle(buffers.seq_lens),
            q_start: handle(buffers.q_start),
            q_len: handle(buffers.q_len),
            prefix_len: handle(buffers.prefix_len),
            live_batch: buffers.live_batch.handle,
            k_new: handle(buffers.k_new),
            v_new: handle(buffers.v_new),
            sink: handle(buffers.sink),
            output: buffers.output.handle,
            workspace: buffers.workspace.handle,
            k_scale: cache.k_scale,
            v_scale: cache.v_scale,
        };
        let planned = self.plan(choice, grid_rows, live, legacy, fold)?;
        buffers.workspace.covers(planned.partial.workspace_bytes)?;
        let launch = |symbol, plan, arguments| Launch {
            symbol,
            rows,
            plan,
            arguments,
            collective: false,
            persistent: false,
        };
        let mut launches = vec![launch(
            planned.partial_symbol,
            planned.partial,
            planned.partial_args,
        )];
        if !planned.direct_output {
            launches.push(launch(
                planned.merge_symbol,
                planned.merge,
                planned.merge_args,
            ));
        }
        Ok(launches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Profile;
    use crate::attention::{EXACT_ROWS, Shape, variant};
    use crate::backend::select::orders;
    use crate::backend::{Sm120, view};
    use kernel_core::attention::{FoldInputs, KvCache};
    use kernel_core::backend::Backend;
    use kernel_core::contract::AttentionPhase::{Decode, Prefill, Verify};

    const PAGED: Shape = Shape {
        q_heads: 16,
        kv_heads: 1,
        qk_dim: 192,
        v_dim: 128,
        page_size: 64,
        kv_dtype: DType::Bf16,
        layout: CacheLayout::Paged,
        window_left: None,
        sink: false,
        softcap: None,
        causal: true,
    };
    const RING: Shape = Shape {
        q_heads: 16,
        kv_heads: 2,
        page_size: 128,
        layout: CacheLayout::Ring,
        window_left: Some(127),
        sink: true,
        ..PAGED
    };
    const GATED: Shape = Shape {
        q_heads: 6,
        qk_dim: 256,
        v_dim: 256,
        ..PAGED
    };
    const GATED_FOLD: GatedFold = GatedFold {
        rotary: 64,
        sections: [11, 11, 10],
        eps: 1e-6,
    };
    /// Bidirectional block attention over a 2048-slot ring, as a drafter
    /// runs it.
    const BLOCK: Shape = Shape {
        q_heads: 8,
        kv_heads: 1,
        qk_dim: 128,
        v_dim: 128,
        page_size: 2048,
        window_left: Some(1023),
        causal: false,
        ..RING
    };
    const STD: Profile = Profile::StdDenseMoe;

    /// Prefill, a decode that folds RoPE and has exact rows entries, and
    /// verify, for both layouts and a gated decode.
    fn variants() -> Vec<Variant> {
        let folded = |v: Variant| Variant {
            rope_fold: Some(64),
            exact_rows: true,
            ..v
        };
        let prefill = |v: Variant| Variant {
            invariance: Invariance::PREFILL,
            ..v
        };
        vec![
            prefill(variant("paged_prefill", STD, Prefill, PAGED)),
            folded(variant("paged_decode", STD, Decode, PAGED)),
            variant("paged_verify", STD, Verify, PAGED),
            prefill(variant("ring_prefill", STD, Prefill, RING)),
            folded(variant("ring_decode", STD, Decode, RING)),
            variant("ring_verify", STD, Verify, RING),
            Variant {
                gated_fold: Some(GATED_FOLD),
                ..variant("gated_decode", STD, Decode, GATED)
            },
            variant("block_verify", STD, Verify, BLOCK),
        ]
    }

    fn backend() -> Sm120 {
        Sm120::new().with_attention(AttentionEntries {
            variants: variants(),
            ..AttentionEntries::default()
        })
    }

    fn named(name: &str) -> Variant {
        variants().into_iter().find(|v| v.name == name).unwrap()
    }

    fn op(v: Variant, fold: Option<AttentionFold>) -> AttentionOp {
        AttentionOp {
            query: v.query,
            item_rows: v.query_capacity,
            fold,
            plan_at_capacity: false,
        }
    }

    const NEOX: AttentionFold = AttentionFold::Neox {
        rotary: 64,
        theta: 10_000.0,
        v_multiplier: 0.5,
        source: RotarySource::Computed,
    };

    const BIG: u64 = 1 << 40;

    fn buffers(v: &Variant) -> AttentionBuffers<View> {
        let paged = v.layout == CacheLayout::Paged;
        let some = |handle| Some(view(handle, BIG));
        AttentionBuffers {
            q: view(1, BIG),
            cache: KvCache {
                k: view(2, BIG),
                v: view(3, BIG),
                slots: 40,
                k_slot_stride: slot_bytes(&v.query, v.query.qk_dim),
                v_slot_stride: slot_bytes(&v.query, v.query.v_dim),
                k_scale: 0.5,
                v_scale: 0.25,
            },
            block_tables: paged.then(|| view(4, BIG)),
            slot_ids: (!paged).then(|| view(5, BIG)),
            seq_lens: some(6),
            q_start: some(7),
            q_len: some(8),
            prefix_len: some(9),
            live_batch: view(10, 4),
            k_new: some(11),
            v_new: some(12),
            sink: v.query.sink.then(|| view(13, BIG)),
            rope: Some(FoldInputs {
                cos_sin: some(14),
                q_norm: some(15),
                k_norm: some(16),
                rope_positions: some(17),
            }),
            output: view(18, BIG),
            workspace: view(19, BIG),
        }
    }

    /// The builder's buffers for `b`, with tensor maps in every phase as a
    /// host binds them today.
    fn legacy(v: Variant, b: &AttentionBuffers<View>) -> Buffers {
        let (k_map, v_map) = v
            .cache_maps_with_strides(
                b.cache.slots,
                b.cache.k.handle,
                b.cache.v.handle,
                b.cache.k_slot_stride,
                b.cache.v_slot_stride,
            )
            .unwrap();
        let handle = |view: Option<View>| view.map(|view| view.handle);
        Buffers {
            q: b.q.handle,
            k_cache: b.cache.k.handle,
            v_cache: b.cache.v.handle,
            k_page_stride_bytes: b.cache.k_slot_stride,
            v_page_stride_bytes: b.cache.v_slot_stride,
            k_map: Some(k_map),
            v_map: Some(v_map),
            block_tables: handle(b.block_tables),
            slot_ids: handle(b.slot_ids),
            seq_lens: handle(b.seq_lens),
            q_start: handle(b.q_start),
            q_len: handle(b.q_len),
            prefix_len: handle(b.prefix_len),
            live_batch: b.live_batch.handle,
            k_new: handle(b.k_new),
            v_new: handle(b.v_new),
            sink: handle(b.sink),
            output: b.output.handle,
            workspace: b.workspace.handle,
            k_scale: b.cache.k_scale,
            v_scale: b.cache.v_scale,
        }
    }

    fn expected(planned: PlannedAttention, rows: u32) -> Vec<Launch> {
        let launch = |symbol, plan, arguments| Launch {
            symbol,
            rows,
            plan,
            arguments,
            collective: false,
            persistent: false,
        };
        let mut launches = vec![launch(
            planned.partial_symbol,
            planned.partial,
            planned.partial_args,
        )];
        if !planned.direct_output {
            launches.push(launch(
                planned.merge_symbol,
                planned.merge,
                planned.merge_args,
            ));
        }
        launches
    }

    /// Counts whose rotary positions are the cache positions, as a NeoX
    /// fold's are.
    fn counts(rows: u32, items: u32, context: u32) -> Counts {
        Counts::rows(rows)
            .with_items(items, context)
            .with_positions(context)
    }

    fn counted(counts: Counts) -> RuntimeCounts {
        live(counts.context, counts.rows, counts.items)
    }

    /// Prepares `op` on `backend` and checks its launches at `counts`
    /// against the builder's `legacy` plan.
    fn check(
        backend: &Sm120,
        op: AttentionOp,
        need: Need,
        counts: Counts,
        legacy: impl Fn(Buffers, RuntimeCounts) -> Result<PlannedAttention, Refusal>,
        v: Variant,
    ) -> Prepared<AttentionBuffers<View>> {
        let prepared = backend.attention(&op, need).unwrap();
        let b = buffers(&v);
        let launches = prepared.launches(&b, View::EMPTY, counts).unwrap();
        let planned = legacy(self::legacy(v, &b), counted(counts)).unwrap();
        assert!(planned.partial.workspace_bytes <= prepared.info().bytes_of("workspace").unwrap());
        assert_eq!(launches, expected(planned, counts.rows));
        for launch in &launches {
            assert!(
                prepared
                    .info()
                    .symbols
                    .iter()
                    .any(|(symbol, _)| *symbol == launch.symbol)
            );
        }
        prepared
    }

    #[test]
    fn launched_entries_keep_their_common_guarantee() {
        let declared = |pairs: &[(&str, Invariance)]| {
            pairs
                .iter()
                .map(|&(symbol, invariance)| (symbol.to_owned(), invariance))
                .collect::<Vec<_>>()
        };
        let symbols = |names: &[&str]| names.iter().map(|&n| n.to_owned()).collect::<Vec<_>>();
        let (narrow, wide) = (Invariance::any_rows(8), Invariance::any_rows(32));
        let both = declared(&[("partial", narrow), ("merge", wide)]);
        assert_eq!(
            launched_guarantees(&both, &symbols(&["partial", "merge"])).unwrap(),
            narrow
        );
        // Only the launched symbols count.
        assert_eq!(
            launched_guarantees(&both, &symbols(&["merge"])).unwrap(),
            wide
        );
        // A symbol declared twice claims what both declarations do.
        let twice = declared(&[("partial", wide), ("partial", narrow)]);
        assert_eq!(
            launched_guarantees(&twice, &symbols(&["partial"])).unwrap(),
            narrow
        );
        // An entry without claims leaves none of the rows claims.
        let unclaimed = declared(&[("partial", wide), ("merge", Invariance::NONE)]);
        assert_eq!(
            launched_guarantees(&unclaimed, &symbols(&["partial", "merge"])).unwrap(),
            Invariance::NONE
        );
        assert_eq!(launched_guarantees(&both, &[]).unwrap(), Invariance::NONE);
        assert!(matches!(
            launched_guarantees(&both, &symbols(&["rope"])),
            Err(Refused::Rule(_))
        ));
    }

    #[test]
    fn phase_entries_launch_what_the_builder_plans() {
        let backend = backend();
        for name in ["paged_prefill", "ring_prefill"] {
            let v = named(name);
            // Short contexts write the output directly; longer ones merge.
            for context in [100, 5000] {
                let prepared = check(
                    &backend,
                    op(v, None),
                    Need::rows(8192),
                    counts(300, 32, context),
                    |b, live| v.plan_for_bucket(32, live, b),
                    v,
                );
                assert_eq!(prepared.info().guarantees, Invariance::PREFILL);
            }
        }
        for name in ["paged_decode", "ring_decode"] {
            let v = named(name);
            check(
                &backend,
                op(v, None),
                Need::rows(32),
                counts(4, 3, 4000),
                |b, live| v.plan_for_bucket(4, live, b),
                v,
            );
        }
        for name in ["paged_verify", "ring_verify"] {
            let v = named(name);
            check(
                &backend,
                op(v, None),
                Need::rows(256),
                counts(24, 3, 4000),
                |b, live| v.plan_for_bucket(4, live, b),
                v,
            );
        }
        // Past the catalog's items, the same entries prepared for more.
        let v = named("paged_decode");
        let wide = AttentionOp {
            query: AttentionQuery {
                max_batch_capacity: 64,
                ..v.query
            },
            ..op(v, None)
        };
        let served = v.with_batch_capacity(64).unwrap();
        check(
            &backend,
            wide,
            Need::rows(64),
            counts(64, 40, 4000),
            |b, live| served.plan_for_bucket(64, live, b),
            served,
        );
    }

    #[test]
    fn folds_launch_what_the_builder_plans() {
        let backend = backend();
        let rope = RopeInputs {
            qkv: 1,
            theta: 10_000.0,
            v_multiplier: 0.5,
        };
        for name in ["paged_decode", "ring_decode"] {
            let v = named(name);
            let prepared = check(
                &backend,
                op(v, Some(NEOX)),
                Need::rows(32),
                counts(8, 5, 4000),
                |b, live| v.plan_folded(8, live, b, rope),
                v,
            );
            assert_eq!(prepared.info().guarantees, Invariance::NONE);
            let table = AttentionFold::Neox {
                rotary: 64,
                theta: 10_000.0,
                v_multiplier: 0.5,
                source: RotarySource::CosSinTable,
            };
            check(
                &backend,
                op(v, Some(table)),
                Need::rows(32),
                counts(8, 5, 4000),
                |b, live| v.plan_folded_table(8, live, b, rope, 14),
                v,
            );
        }
        let v = named("gated_decode");
        let fold = AttentionFold::Gated {
            rotary: 64,
            sections: [11, 11, 10],
            eps: 1e-6,
        };
        let inputs = GatedRopeInputs {
            qkv: 1,
            q_norm: 15,
            k_norm: 16,
            rope_positions: 17,
            cos_sin: 14,
        };
        check(
            &backend,
            op(v, Some(fold)),
            Need::rows(32),
            counts(2, 2, 4000),
            |b, live| v.plan_folded_gated(2, live, b, inputs),
            v,
        );
    }

    /// One-row bits of up to `max` verify rows, without batch claims.
    fn one_row(max: u32) -> Invariance {
        Invariance {
            rows: Rows::OneRow { max },
            ..Invariance::NONE
        }
    }

    #[test]
    fn exact_verify_runs_the_decode_rows_entries() {
        let exact = Need::rows(8).with_exact(one_row(EXACT_ROWS));
        let paged = named("paged_decode");
        let verify = op(named("paged_verify"), None);
        let prepared = check(
            &backend(),
            verify,
            exact,
            counts(8, 3, 4000),
            |b, live| paged.plan_rows_shared(8, live, b),
            paged,
        );
        assert_eq!(prepared.info().guarantees, one_row(EXACT_ROWS));
        assert_eq!(
            prepared.info().symbols,
            [
                (
                    paged.rows_shared_symbol(),
                    paged
                        .dynamic_shared_bytes(&paged.rows_shared_symbol())
                        .unwrap()
                ),
                (paged.rows_merge_symbol(), 0)
            ]
        );
        let per_row = Sm120::new().with_attention(AttentionEntries {
            variants: variants(),
            per_row_exact: true,
            rows_above: None,
        });
        check(
            &per_row,
            verify,
            exact,
            counts(8, 3, 4000),
            |b, live| paged.plan_rows(8, live, b),
            paged,
        );
        let ring = named("ring_decode");
        check(
            &backend(),
            op(named("ring_verify"), None),
            exact,
            counts(5, 2, 4000),
            |b, live| ring.plan_rows(5, live, b),
            ring,
        );
        // The diagnostic runs wide verify steps on the per-row entries.
        let rows_above = Sm120::new().with_attention(AttentionEntries {
            variants: variants(),
            per_row_exact: false,
            rows_above: Some(EXACT_ROWS),
        });
        check(
            &rows_above,
            verify,
            Need::rows(32),
            counts(32, 4, 4000),
            |b, live| paged.plan_rows(32, live, b),
            paged,
        );
    }

    /// A step of fewer query rows than the bucket its rows paths grid
    /// passes the bucket as `Counts::padded_rows`: each row path spreads
    /// its grid over the bucket, plans for the step's rows, and reports
    /// them; other paths read no padded rows.
    #[test]
    fn rows_paths_grid_a_steps_padded_bucket() {
        let exact = Need::rows(8).with_exact(one_row(EXACT_ROWS));
        let padded = |rows, items, bucket| Counts {
            padded_rows: bucket,
            ..counts(rows, items, 4000)
        };
        let paged = named("paged_decode");
        let verify = op(named("paged_verify"), None);
        check(
            &backend(),
            verify,
            exact,
            padded(5, 3, 8),
            |b, live| paged.plan_rows_shared(8, live, b),
            paged,
        );
        let per_row = Sm120::new().with_attention(AttentionEntries {
            variants: variants(),
            per_row_exact: true,
            rows_above: None,
        });
        check(
            &per_row,
            verify,
            exact,
            padded(5, 3, 8),
            |b, live| paged.plan_rows(8, live, b),
            paged,
        );
        let ring = named("ring_decode");
        check(
            &backend(),
            op(named("ring_verify"), None),
            exact,
            padded(3, 2, 4),
            |b, live| ring.plan_rows(4, live, b),
            ring,
        );
        let rows_above = Sm120::new().with_attention(AttentionEntries {
            variants: variants(),
            per_row_exact: false,
            rows_above: Some(EXACT_ROWS),
        });
        check(
            &rows_above,
            verify,
            Need::rows(32),
            padded(20, 4, 32),
            |b, live| paged.plan_rows(32, live, b),
            paged,
        );
        // A padded bucket past the prepared rows is refused.
        let prepared = backend().attention(&verify, exact).unwrap();
        assert!(
            prepared
                .launches(&buffers(&paged), View::EMPTY, padded(5, 3, 16))
                .is_err()
        );
        // The verify bucket's own plan does not read padded rows.
        let prepared = backend().attention(&verify, Need::rows(8)).unwrap();
        let b = buffers(&paged);
        assert_eq!(
            prepared.launches(&b, View::EMPTY, padded(5, 3, 8)).unwrap(),
            prepared
                .launches(&b, View::EMPTY, counts(5, 3, 4000))
                .unwrap()
        );
    }

    #[test]
    fn prepare_refuses_unserved_inexact_and_over_limit_ops() {
        let backend = backend();
        let decode = op(named("paged_decode"), None);
        let rule = |result: Result<Prepared<AttentionBuffers<View>>, Refused>| {
            matches!(result, Err(Refused::Rule(_)))
        };
        assert_eq!(
            Sm120::new().attention(&decode, Need::rows(4)).err(),
            Some(Refused::Unsupported)
        );
        // Ops no variant serves: another softcap, fold or verify width.
        let softcap = AttentionOp {
            query: AttentionQuery {
                softcap: Some(30.0),
                ..decode.query
            },
            ..decode
        };
        assert!(rule(backend.attention(&softcap, Need::rows(4))));
        let narrow = AttentionFold::Neox {
            rotary: 32,
            theta: 10_000.0,
            v_multiplier: 0.5,
            source: RotarySource::Computed,
        };
        assert!(rule(backend.attention(
            &op(named("paged_decode"), Some(narrow)),
            Need::rows(4)
        )));
        let verify = op(named("paged_verify"), None);
        assert!(rule(backend.attention(
            &AttentionOp {
                item_rows: 9,
                ..verify
            },
            Need::rows(9)
        )));
        // SM120's head widths and BF16 rings, which the contract leaves
        // open.
        let wide = AttentionOp {
            query: AttentionQuery {
                qk_dim: 96,
                ..decode.query
            },
            ..decode
        };
        let fp8_ring = AttentionOp {
            query: AttentionQuery {
                kv_dtype: DType::Fp8E4M3,
                ..named("ring_decode").query
            },
            ..decode
        };
        for limited in [wide, fp8_ring] {
            assert_eq!(limited.validate(), Ok(()));
            assert!(matches!(
                backend.attention(&limited, Need::rows(4)),
                Err(Refused::Rule(rule))
                    if rule == "unsupported attention head dimension" || rule == "ring KV requires BF16"
            ));
        }
        // The builders keep refusing them.
        let odd = Shape {
            qk_dim: 96,
            ..PAGED
        };
        let fp8_ring = Shape {
            kv_dtype: DType::Fp8E4M3,
            ..RING
        };
        for (shape, refusal) in [
            (odd, "unsupported attention head dimension"),
            (fp8_ring, "ring KV requires BF16"),
        ] {
            let variant = variant("limited", STD, Decode, shape);
            assert_eq!(variant.validate(), Err(Refusal(refusal)));
        }
        // Past SM120's 256-item tables and the variant's rows.
        let huge = AttentionOp {
            query: AttentionQuery {
                max_batch_capacity: 512,
                ..decode.query
            },
            ..decode
        };
        assert!(rule(backend.attention(&huge, Need::rows(512))));
        assert!(rule(backend.attention(&decode, Need::rows(33))));
        assert!(rule(backend.attention(&verify, Need::rows(257))));
        // An oversized capacity is refused before anything is planned.
        let prefill = op(named("paged_prefill"), None);
        for (op, exact) in [
            (decode, Invariance::NONE),
            (verify, Invariance::NONE),
            (verify, one_row(EXACT_ROWS)),
            (prefill, Invariance::NONE),
        ] {
            assert!(rule(
                backend.attention(&op, Need::rows(u32::MAX).with_exact(exact))
            ));
        }
        // Exactness past the declarations: decode and verify entries claim
        // none, rows entries one-row bits of up to eight rows without batch
        // independence, and the shared-key entries hold 16 rows.
        assert!(matches!(
            backend.attention(&decode, Need::rows(4).with_exact(Invariance::one_row(1))),
            Err(Refused::Inexact { .. })
        ));
        for exact in [Invariance::one_row(8), one_row(16)] {
            assert!(matches!(
                backend.attention(&verify, Need::rows(16).with_exact(exact)),
                Err(Refused::Inexact { .. })
            ));
        }
        assert!(rule(
            backend.attention(&verify, Need::rows(32).with_exact(one_row(8)))
        ));
        assert!(
            backend
                .attention(&prefill, Need::rows(64).with_exact(Invariance::PREFILL))
                .is_ok()
        );
    }

    #[test]
    fn launches_refuse_short_views_and_counts_past_capacity() {
        let v = named("paged_decode");
        let prepared = backend().attention(&op(v, None), Need::rows(16)).unwrap();
        let b = buffers(&v);
        assert!(
            prepared
                .launches(&b, View::EMPTY, counts(16, 16, 4000))
                .is_ok()
        );
        for counts in [
            counts(0, 1, 4000),
            counts(17, 16, 4000),
            counts(4, 0, 4000),
            counts(4, 33, 4000),
            counts(4, 4, 0),
        ] {
            assert!(prepared.launches(&b, View::EMPTY, counts).is_err());
        }
        // The step's own workspace, at most the largest bucket's.
        let workspace = prepared
            .launches(&b, View::EMPTY, counts(16, 16, 4000))
            .unwrap()[0]
            .plan
            .workspace_bytes;
        assert!(workspace <= prepared.info().bytes_of("workspace").unwrap());
        let rows = 16 * 16 * 128 * 2;
        let short = [
            AttentionBuffers {
                output: view(18, rows - 2),
                ..b
            },
            AttentionBuffers {
                workspace: view(19, workspace - 4),
                ..b
            },
            AttentionBuffers {
                block_tables: Some(view(4, 16 * 1_048_576 / 64 * 4 - 4)),
                ..b
            },
            AttentionBuffers {
                cache: KvCache {
                    k: view(2, 39 * b.cache.k_slot_stride),
                    ..b.cache
                },
                ..b
            },
        ];
        let full = AttentionBuffers {
            output: view(18, rows),
            workspace: view(19, workspace),
            ..b
        };
        assert!(
            prepared
                .launches(&full, View::EMPTY, counts(16, 16, 4000))
                .is_ok()
        );
        for short in short {
            assert!(
                prepared
                    .launches(&short, View::EMPTY, counts(16, 16, 4000))
                    .is_err()
            );
        }
        // A fold takes its inputs.
        let table = AttentionFold::Neox {
            rotary: 64,
            theta: 10_000.0,
            v_multiplier: 0.5,
            source: RotarySource::CosSinTable,
        };
        let folded = backend()
            .attention(&op(v, Some(table)), Need::rows(16))
            .unwrap();
        assert!(folded.launches(&b, View::EMPTY, counts(4, 4, 4000)).is_ok());
        let bare = AttentionBuffers { rope: None, ..b };
        assert!(
            folded
                .launches(&bare, View::EMPTY, counts(4, 4, 4000))
                .is_err()
        );
    }

    /// R1 review regressions.

    #[test]
    fn ring_slots_are_tightly_packed() {
        let v = named("ring_decode");
        let prepared = backend().attention(&op(v, None), Need::rows(4)).unwrap();
        let b = buffers(&v);
        let ring = b.cache.k_slot_stride;
        // Two rings over planes that hold one, whatever strides claim.
        let one_ring = |k_slot_stride, v_slot_stride| AttentionBuffers {
            cache: KvCache {
                k: view(2, ring),
                v: view(3, b.cache.v_slot_stride),
                slots: 2,
                k_slot_stride,
                v_slot_stride,
                ..b.cache
            },
            ..b
        };
        for strides in [(0, 0), (ring, b.cache.v_slot_stride), (ring * 2, 0)] {
            let short = one_ring(strides.0, strides.1);
            assert!(
                prepared
                    .launches(&short, View::EMPTY, counts(2, 2, 100))
                    .is_err()
            );
        }
        let two = AttentionBuffers {
            cache: KvCache {
                k: view(2, 2 * ring),
                v: view(3, 2 * b.cache.v_slot_stride),
                slots: 2,
                ..b.cache
            },
            ..b
        };
        assert!(
            prepared
                .launches(&two, View::EMPTY, counts(2, 2, 100))
                .is_ok()
        );
    }

    #[test]
    fn rope_tables_cover_every_position_of_the_step() {
        let v = named("paged_decode");
        let table = AttentionFold::Neox {
            rotary: 64,
            theta: 10_000.0,
            v_multiplier: 0.5,
            source: RotarySource::CosSinTable,
        };
        let prepared = backend()
            .attention(&op(v, Some(table)), Need::rows(4))
            .unwrap();
        let b = buffers(&v);
        let with_table = |bytes| AttentionBuffers {
            rope: Some(FoldInputs {
                cos_sin: Some(view(14, bytes)),
                ..b.rope.unwrap()
            }),
            ..b
        };
        // Context 4 reads positions 0 to 3: 64 rotary dimensions of F32
        // cos and sin pairs, 256 bytes each.
        assert!(
            prepared
                .launches(&with_table(256), View::EMPTY, counts(1, 1, 4))
                .is_err()
        );
        assert!(
            prepared
                .launches(&with_table(1023), View::EMPTY, counts(1, 1, 4))
                .is_err()
        );
        assert!(
            prepared
                .launches(&with_table(1024), View::EMPTY, counts(1, 1, 4))
                .is_ok()
        );
        let gated = named("gated_decode");
        let fold = AttentionFold::Gated {
            rotary: 64,
            sections: [11, 11, 10],
            eps: 1e-6,
        };
        let prepared = backend()
            .attention(&op(gated, Some(fold)), Need::rows(4))
            .unwrap();
        let b = buffers(&gated);
        let short = AttentionBuffers {
            rope: Some(FoldInputs {
                cos_sin: Some(view(14, 1023)),
                ..b.rope.unwrap()
            }),
            ..b
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, counts(1, 1, 4))
                .is_err()
        );
        // A gated fold's M-RoPE positions need not be cache positions: the
        // table covers the step's rotary positions, whatever its context.
        let positions = |positions| counts(1, 1, 4).with_positions(positions);
        assert!(prepared.launches(&short, View::EMPTY, positions(3)).is_ok());
        assert!(
            prepared
                .launches(&short, View::EMPTY, positions(5))
                .is_err()
        );
    }

    #[test]
    fn the_wide_verify_diagnostic_leaves_block_attention_alone() {
        let rows_above = Sm120::new().with_attention(AttentionEntries {
            variants: variants(),
            per_row_exact: false,
            rows_above: Some(EXACT_ROWS),
        });
        let block = named("block_verify");
        check(
            &rows_above,
            op(block, None),
            Need::rows(32),
            counts(32, 4, 4000),
            |b, live| block.plan_for_bucket(4, live, b),
            block,
        );
        // Exactness it cannot give is still refused.
        assert!(matches!(
            rows_above.attention(&op(block, None), Need::rows(8).with_exact(one_row(8))),
            Err(Refused::Rule(_))
        ));
    }

    #[test]
    fn every_bucket_launches_its_prepared_choice() {
        let backend = backend();
        for name in ["paged_decode", "ring_decode", "paged_verify", "ring_verify"] {
            let v = named(name);
            let prepared = backend.attention(&op(v, None), Need::rows(32)).unwrap();
            let b = buffers(&v);
            for items in 1..=32 {
                let rows = if v.query.phase == Decode {
                    items
                } else {
                    32.min(items * 8)
                };
                let counts = counts(rows, items, 4000);
                let bucket = items.next_power_of_two();
                let planned = v
                    .plan_for_bucket(bucket, counted(counts), legacy(v, &b))
                    .unwrap();
                assert_eq!(
                    prepared.launches(&b, View::EMPTY, counts).unwrap(),
                    expected(planned, rows),
                    "{name} at {items} items"
                );
            }
        }
    }

    #[test]
    fn launches_keep_the_ops_own_item_limit() {
        let v = named("paged_verify");
        let single = AttentionOp {
            query: AttentionQuery {
                max_batch_capacity: 1,
                ..v.query
            },
            ..op(v, None)
        };
        let prepared = backend().attention(&single, Need::rows(8)).unwrap();
        let b = buffers(&v);
        assert!(
            prepared
                .launches(&b, View::EMPTY, counts(2, 2, 4000))
                .is_err()
        );
        assert!(
            prepared
                .launches(&b, View::EMPTY, counts(8, 1, 4000))
                .is_ok()
        );
        assert!(backend().attention(&single, Need::rows(9)).is_err());
    }

    /// An op planned at its capacity checks the per-item arrays at the
    /// step's items and plans as for its prepared items, as a drafter's
    /// block attention replays one graph whatever items a step drafts.
    #[test]
    fn ops_planned_at_capacity_plan_their_items_over_the_steps_arrays() {
        let v = named("block_verify");
        let items = v.query.max_batch_capacity;
        let fixed = AttentionOp {
            plan_at_capacity: true,
            ..op(v, None)
        };
        let prepared = backend().attention(&fixed, Need::rows(16)).unwrap();
        // Per-item arrays of two items.
        let mut b = buffers(&v);
        for (handle, array) in [
            (5, &mut b.slot_ids),
            (6, &mut b.seq_lens),
            (7, &mut b.q_start),
            (8, &mut b.q_len),
            (9, &mut b.prefix_len),
        ] {
            *array = Some(view(handle, 2 * 4));
        }
        let at = counts(16, 2, 4000);
        let planned = v
            .plan_for_bucket(items, live(4000, 16, items), legacy(v, &b))
            .unwrap();
        assert_eq!(
            prepared.launches(&b, View::EMPTY, at).unwrap(),
            expected(planned, 16)
        );
        // The arrays are checked at the step's items.
        assert!(
            prepared
                .launches(&b, View::EMPTY, counts(16, 3, 4000))
                .is_err()
        );
        // Planned for the step's items, the same step launches their
        // bucket's plan.
        let step = backend().attention(&op(v, None), Need::rows(16)).unwrap();
        let planned = v
            .plan_for_bucket(2, live(4000, 16, 2), legacy(v, &b))
            .unwrap();
        assert_eq!(
            step.launches(&b, View::EMPTY, at).unwrap(),
            expected(planned, 16)
        );
        assert_ne!(
            step.launches(&b, View::EMPTY, at).unwrap(),
            prepared.launches(&b, View::EMPTY, at).unwrap()
        );
    }

    #[test]
    fn decode_refuses_more_items_than_rows() {
        let v = named("paged_decode");
        let prepared = backend().attention(&op(v, None), Need::rows(1)).unwrap();
        let b = buffers(&v);
        assert!(prepared.launches(&b, View::EMPTY, counts(1, 2, 4)).is_err());
        assert!(prepared.launches(&b, View::EMPTY, counts(1, 1, 4)).is_ok());
    }

    #[test]
    fn verify_items_may_hold_fewer_rows_than_the_entry() {
        let v = named("paged_verify");
        let short = AttentionOp {
            item_rows: 4,
            ..op(v, None)
        };
        check(
            &backend(),
            short,
            Need::rows(16),
            counts(16, 4, 4000),
            |b, live| v.plan_for_bucket(4, live, b),
            v,
        );
        // Four-row items of 32 hold at most 128 rows.
        assert!(backend().attention(&short, Need::rows(129)).is_err());
    }

    /// R2 review regression.

    #[test]
    fn capacities_hold_the_padded_rows_of_the_items_bucket() {
        let three = |v: Variant| AttentionOp {
            query: AttentionQuery {
                max_batch_capacity: 3,
                ..v.query
            },
            ..op(v, None)
        };
        // Three items pad to a bucket of four decode rows, or of 32 verify
        // rows of eight per item; the step keeps its padded rows.
        let decode = named("paged_decode");
        check(
            &backend(),
            three(decode),
            Need::rows(4),
            counts(4, 3, 4000),
            |b, live| decode.plan_for_bucket(4, live, b),
            decode,
        );
        let verify = named("paged_verify");
        check(
            &backend(),
            three(verify),
            Need::rows(32),
            counts(32, 3, 4000),
            |b, live| verify.plan_for_bucket(4, live, b),
            verify,
        );
        // The item limit still holds at launch, and past the bucket rows.
        let prepared = backend().attention(&three(decode), Need::rows(4)).unwrap();
        assert!(
            prepared
                .launches(&buffers(&decode), View::EMPTY, counts(4, 4, 4000))
                .is_err()
        );
        assert!(backend().attention(&three(decode), Need::rows(5)).is_err());
        assert!(backend().attention(&three(verify), Need::rows(33)).is_err());
    }

    #[test]
    fn variants_prepare_alike_in_every_order() {
        let decode = named("paged_decode");
        let verify = named("paged_verify");
        let prepare = |order: &[Variant], op: AttentionOp, need: Need| {
            Sm120::new()
                .with_attention(AttentionEntries {
                    variants: order.to_vec(),
                    ..AttentionEntries::default()
                })
                .attention(&op, need)
                .map(|prepared| prepared.info().clone())
        };
        let in_every_order = |registered: &[Variant], op, need| {
            let results = orders(registered)
                .iter()
                .map(|order| prepare(order, op, need))
                .collect::<Vec<_>>();
            assert!(results.iter().all(|result| *result == results[0]));
            results[0].clone()
        };
        let ordinary = Need::rows(8);
        let exact = Need::rows(8).with_exact(one_row(EXACT_ROWS));
        // A duplicate registration is one variant, for the phase entries
        // and for exact verify rows.
        let registered = [decode, verify, decode, verify];
        for need in [ordinary, exact] {
            assert_eq!(
                in_every_order(&registered, op(verify, None), need),
                prepare(&[decode, verify], op(verify, None), need),
            );
        }
        // A variant under another name ties, for the op whose entries it
        // provides.
        let tied = |result: Result<OpInfo, Refused>, mut names: [&str; 2]| {
            names.sort_unstable();
            matches!(result, Err(Refused::Rule(rule)) if rule.contains(&names.join(" and ")))
        };
        let twin = Variant {
            name: "paged_verify_twin",
            ..verify
        };
        let registered = [decode, verify, twin];
        assert!(tied(
            in_every_order(&registered, op(verify, None), ordinary),
            [verify.name, twin.name]
        ));
        assert!(in_every_order(&registered, op(verify, None), exact).is_ok());
        let twin = Variant {
            name: "paged_decode_twin",
            ..decode
        };
        let registered = [decode, verify, twin];
        assert!(in_every_order(&registered, op(verify, None), ordinary).is_ok());
        assert!(tied(
            in_every_order(&registered, op(verify, None), exact),
            [decode.name, twin.name]
        ));
        // The least query capacity that holds the item rows wins.
        let narrow = Variant {
            name: "paged_verify_narrow",
            query_capacity: 4,
            ..verify
        };
        let short = AttentionOp {
            item_rows: 4,
            ..op(verify, None)
        };
        let registered = [verify, narrow];
        let narrow_info = prepare(&[narrow], short, Need::rows(16)).unwrap();
        assert_eq!(
            in_every_order(&registered, short, Need::rows(16)),
            Ok(narrow_info)
        );
        let wide_info = prepare(&[verify], op(verify, None), Need::rows(16)).unwrap();
        assert_eq!(
            in_every_order(&registered, op(verify, None), Need::rows(16)),
            Ok(wide_info)
        );
    }

    /// Prepares `op` at `need` over every order of `registered`: the one
    /// result all orders give.
    fn alike(registered: &[Variant], op: AttentionOp, need: Need) -> Result<OpInfo, Refused> {
        let results = orders(registered)
            .iter()
            .map(|order| {
                Sm120::new()
                    .with_attention(AttentionEntries {
                        variants: order.clone(),
                        ..AttentionEntries::default()
                    })
                    .attention(&op, need)
                    .map(|prepared| prepared.info().clone())
            })
            .collect::<Vec<_>>();
        assert!(results.iter().all(|result| *result == results[0]));
        results[0].clone()
    }

    /// R1 finding 3: a variant that cannot keep the needed exactness ties
    /// with nothing; R1 finding 4: a variant that differs only in its claim
    /// launches the same and is one variant.
    #[test]
    fn exactness_filters_variants_before_ties() {
        let prefill = named("paged_prefill");
        assert_eq!(prefill.invariance, Invariance::PREFILL);
        let unclaimed = Variant {
            name: "paged_prefill_twin",
            invariance: Invariance::NONE,
            ..prefill
        };
        let exact = Need::rows(16).with_exact(Invariance::PREFILL);
        let alone = alike(&[prefill], op(prefill, None), exact).unwrap();
        assert_eq!(
            alike(&[prefill, unclaimed], op(prefill, None), exact),
            Ok(alone)
        );
        assert!(matches!(
            alike(&[prefill, unclaimed], op(prefill, None), Need::rows(16)),
            Err(Refused::Rule(rule)) if rule.contains("paged_prefill and paged_prefill_twin")
        ));
        let claim_only = Variant {
            invariance: Invariance::NONE,
            ..prefill
        };
        let plain = alike(&[claim_only], op(prefill, None), Need::rows(16)).unwrap();
        let both = alike(&[prefill, claim_only], op(prefill, None), Need::rows(16)).unwrap();
        assert_eq!(both.symbols, plain.symbols);
        assert_eq!(
            alike(&[prefill, claim_only], op(prefill, None), exact),
            Ok(alike(&[prefill], op(prefill, None), exact).unwrap())
        );
    }

    /// R1 finding 5: of variants that hold an op's items, the least batch
    /// capacity runs it; variants that hold fewer are prepared for the op's
    /// items after those, and then launch alike.
    #[test]
    fn the_least_batch_capacity_that_holds_the_items_wins() {
        let verify = named("paged_verify");
        let four = verify.with_batch_capacity(4).unwrap();
        let eight = verify.with_batch_capacity(8).unwrap();
        let at = |items| AttentionOp {
            query: AttentionQuery {
                max_batch_capacity: items,
                ..verify.query
            },
            ..op(verify, None)
        };
        let need = Need::rows(16);
        for (items, holder) in [(4, four), (6, eight)] {
            let single = alike(&[holder], at(items), need).unwrap();
            assert_eq!(alike(&[four, eight], at(items), need), Ok(single));
        }
        let single = alike(&[four], at(16), need).unwrap();
        assert_eq!(alike(&[four, eight], at(16), need), Ok(single));
        // Under two names, variants prepared for the op's items still tie.
        let other = Variant {
            name: "paged_verify_twin",
            ..eight
        };
        assert!(matches!(
            alike(&[four, other], at(16), need),
            Err(Refused::Rule(rule)) if rule.contains("paged_verify and paged_verify_twin")
        ));
        assert_eq!(
            alike(&[four, other], at(4), need),
            Ok(alike(&[four], at(4), need).unwrap())
        );
    }

    /// R2 finding 2: decode variants that differ only in whether they also
    /// have exact verify rows entries launch the same decode, so they are
    /// one variant for a decode op in every order; an exact verify op takes
    /// the one that has the rows entries.
    #[test]
    fn variants_of_one_launch_on_the_path_are_one() {
        let decode = named("paged_decode");
        assert!(decode.exact_rows);
        let plain = Variant {
            exact_rows: false,
            ..decode
        };
        let launches = |registered: &[Variant], op: AttentionOp, need: Need, at: Counts| {
            let results = orders(registered)
                .iter()
                .map(|order| {
                    Sm120::new()
                        .with_attention(AttentionEntries {
                            variants: order.clone(),
                            ..AttentionEntries::default()
                        })
                        .attention(&op, need)
                        .map(|prepared| {
                            let info = prepared.info().clone();
                            (
                                info,
                                prepared
                                    .launches(&buffers(&decode), View::EMPTY, at)
                                    .unwrap(),
                            )
                        })
                })
                .collect::<Vec<_>>();
            assert!(results.iter().all(|result| *result == results[0]));
            results[0].clone()
        };
        for context in [1, 128, 4096] {
            let at = counts(1, 1, context);
            let (_, alone) = launches(&[plain], op(decode, None), Need::rows(1), at).unwrap();
            let (_, both) =
                launches(&[decode, plain], op(decode, None), Need::rows(1), at).unwrap();
            assert_eq!(both, alone, "context {context}");
        }
        let verify = op(named("paged_verify"), None);
        let exact = Need::rows(8).with_exact(one_row(EXACT_ROWS));
        let at = counts(8, 3, 4000);
        assert_eq!(
            launches(&[decode, plain, named("paged_verify")], verify, exact, at),
            launches(&[decode, named("paged_verify")], verify, exact, at)
        );
    }

    /// The attention key holds what a variant launches on its path, every
    /// field changed alone: the entries its name gives and its row
    /// capacity; the folds and exact rows entries another path launches,
    /// its profile and its claims are not facts of the phase path. The
    /// query, layout and folds select the op.
    #[test]
    fn launch_keys_hold_what_the_path_launches() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let other = Profile::synthetic(crate::ProfileNames {
            name: "attention-other",
            tag: "attention_other",
        });
        let decode = named("paged_decode");
        let names = field_names!(Variant {
            name,
            profile,
            query,
            layout,
            query_capacity,
            rope_fold,
            gated_fold,
            invariance,
            exact_rows
        });
        let fields: Vec<Field<Variant>> = vec![
            (
                "name",
                Class::Launch,
                Box::new(|v| v.name = "paged_decode_twin"),
            ),
            (
                "profile",
                Class::Inert,
                Box::new(move |v| v.profile = other),
            ),
            ("query", Class::Selects, Box::new(|v| v.query.q_heads *= 2)),
            (
                "layout",
                Class::Selects,
                Box::new(|v| v.layout = CacheLayout::Ring),
            ),
            (
                "query_capacity",
                Class::Selects,
                Box::new(|v| v.query_capacity = 2),
            ),
            (
                "rope_fold",
                Class::Selects,
                Box::new(|v| v.rope_fold = Some(32)),
            ),
            (
                "gated_fold",
                Class::Selects,
                Box::new(|v| {
                    v.gated_fold = Some(GatedFold {
                        rotary: 64,
                        sections: [11, 11, 10],
                        eps: 1e-6,
                    })
                }),
            ),
            (
                "invariance",
                Class::Inert,
                Box::new(|v| v.invariance = Invariance::PREFILL),
            ),
            (
                "exact_rows",
                Class::Inert,
                Box::new(|v| v.exact_rows = !v.exact_rows),
            ),
        ];
        let at = [1, 128, 4096].map(|context| counts(1, 1, context));
        keys_follow_launches(
            &decode,
            names,
            fields,
            |v| launch(v, Path::Phase),
            |v| {
                let prepared = Sm120::new()
                    .with_attention(AttentionEntries {
                        variants: vec![*v],
                        ..AttentionEntries::default()
                    })
                    .attention(&op(decode, None), Need::rows(1))?;
                let launches = at.map(|at| {
                    prepared
                        .launches(&buffers(&decode), View::EMPTY, at)
                        .unwrap()
                });
                Ok((String::new(), launches))
            },
        );
    }
}
