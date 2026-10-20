//! The row-local ops other than RMSNorm through the entries a model
//! instantiates: the row engine's (`row::RowInstantiation`: embedding,
//! logits argmax and top-k, sharded argmax), the shared sharded argmax
//! (`row::argmax`) and the hyper-connection rows
//! (`row::hyper_connection::Entry`: grouped norm, bottleneck, mix, inject,
//! n-gram hash gather and PLE gate). Each prepared op launches its entry
//! once per step, planned as the model's builder plans it.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{
    ArgmaxLayout, Argument, ArgumentList, DType, LaunchPlan, NgramQuery, Refusal, RowQuery,
    RuntimeCounts, SigmoidMeanHyperConnectionQuery as H, StaticQuery, Validate,
};
use kernel_core::properties::Invariance;
use kernel_core::rows::{
    ArgmaxFinalBuffers, ArgmaxPairs, ArgmaxPartialBuffers, BottleneckBuffers, EmbeddingBuffers,
    GroupedNormBuffers, InjectBuffers, LogitsBuffers, MixBuffers, NgramBuffers, PleGateBuffers,
};

use super::select::{Candidate, Chosen, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::comm::epilogue::{GroupedRmsNorm, HC_INJECT_NORM_EPS_BITS};
use crate::row::argmax;
use crate::row::hyper_connection::Entry;
use crate::row::{RowInstantiation, smallest_bucket};

/// A row engine entry of the model's selection and whether it is a
/// sharded embedding, which subtracts its table's first vocabulary row.
pub(super) type EngineRow = (RowInstantiation, bool);

/// The row entries of a model's selection.
#[derive(Default)]
pub(super) struct Selection {
    pub(super) engine: Vec<EngineRow>,
    pub(super) hyper: Vec<Entry>,
    /// Grouped norm entries by symbol, which serve hyper-connection grouped
    /// norms over their weight dtype.
    pub(super) grouped_norms: Vec<(GroupedRmsNorm, String)>,
    /// n-gram hash entries and the end-of-document token each compiles in.
    pub(super) ngrams: Vec<(Entry, u32)>,
    /// Shared sharded argmax entries (`row::argmax`).
    pub(super) argmax: Vec<argmax::Entry>,
}

/// The versioned implementation of an entry: its symbol and a digest of
/// its whole declaration (query, body, grid and threads), which fixes its
/// numerics.
fn implementation(symbol: &str, declaration: String) -> String {
    let digest = kernel_build::digest(declaration.as_bytes());
    format!("sm120.row.v1[{symbol}]#{}", &digest[..16])
}

fn dense(rows: u32, elements: u64, bytes: u64) -> Result<u64, Refusal> {
    u64::from(rows)
        .checked_mul(elements)
        .and_then(|n| n.checked_mul(bytes))
        .ok_or(Refusal("row buffer bytes overflow"))
}

fn buffer(view: View) -> Argument {
    Argument::DeviceBuffer(view.handle)
}

/// One row engine entry launched at the smallest capacity bucket that
/// holds a step's rows, as the row builders plan it: rows at or past the
/// device live count are not written, so every bucket writes equal bits.
pub(super) struct EngineOp {
    entry: RowInstantiation,
    /// The plan of each row count from one to the prepared capacity.
    plans: Vec<LaunchPlan>,
    pub(super) info: OpInfo,
}

impl EngineOp {
    /// `chosen`'s entry, claiming what its registrations declare.
    pub(super) fn new(chosen: Chosen<RowInstantiation>, need: Need) -> Result<Self, Refused> {
        let Chosen { entry, guarantees } = chosen;
        served(need)?;
        if smallest_bucket(need.capacity).is_none() {
            return Err(Refused::Rule(
                "the row capacity exceeds every SM120 row bucket".into(),
            ));
        }
        let plans = plans(need.capacity, |rows| {
            let bucket = smallest_bucket(rows).ok_or(Refusal("no row bucket holds the rows"))?;
            entry.plan(bucket, counts(rows))
        })?;
        let info = OpInfo::new(
            implementation(&entry.symbol, format!("{entry:?}")),
            guarantees,
            vec![(entry.symbol.clone(), 0)],
        );
        need.check(&info)?;
        Ok(Self { entry, plans, info })
    }

    pub(super) fn rows(&self, counts: Counts) -> Result<u32, Refusal> {
        planned(&self.plans, counts).map(|_| counts.rows)
    }

    pub(super) fn launch(
        &self,
        rows: u32,
        arguments: Vec<Argument>,
    ) -> Result<Vec<Launch>, Refusal> {
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows,
            plan: planned(&self.plans, Counts::rows(rows))?,
            arguments: self.entry.pack_arguments(arguments)?,
            collective: false,
            persistent: false,
        }])
    }
}

/// Refuses a capacity no prepared op serves, before any per-row table is
/// built.
fn served(need: Need) -> Result<(), Refused> {
    if need.capacity == 0 || need.capacity > MAX_CAPACITY {
        return Err(Refused::Rule(
            "a prepared op serves 1 to MAX_CAPACITY rows".into(),
        ));
    }
    Ok(())
}

/// The most rows one SM120 row launch serves: the largest row capacity
/// bucket, and the bound every hyper-connection row entry compiles to.
const MAX_ROWS: u32 = crate::row::hyper_connection::MAX_ROWS;

/// The plan of each row count from one to `capacity`, checked against
/// SM120 when the op is prepared, so a launch only indexes it. The
/// capacity bound and the largest plan are checked first, so a capacity
/// the entry refuses never grows the table.
fn plans(
    capacity: u32,
    plan: impl Fn(u32) -> Result<LaunchPlan, Refusal>,
) -> Result<Vec<LaunchPlan>, Refusal> {
    if capacity == 0 || capacity > MAX_ROWS {
        return Err(Refusal("an SM120 row op serves 1 to 8192 rows"));
    }
    let largest = plan(capacity)?;
    check_plan(&largest)?;
    let mut plans = (1..capacity)
        .map(|rows| {
            let plan = plan(rows)?;
            check_plan(&plan)?;
            Ok(plan)
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    plans.push(largest);
    Ok(plans)
}

/// The prepared plan of `counts.rows`.
fn planned(plans: &[LaunchPlan], counts: Counts) -> Result<LaunchPlan, Refusal> {
    counts
        .rows
        .checked_sub(1)
        .and_then(|index| plans.get(index as usize))
        .copied()
        .ok_or(Refusal("row count outside the prepared capacity"))
}

fn counts(rows: u32) -> RuntimeCounts {
    RuntimeCounts {
        m: rows,
        batch: 1,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    }
}

/// The entry of `rows` whose query is `op` among those `rank` accepts
/// (`Some` rank of a sharded flag), the least rank first.
fn engine_entry(
    rows: &[EngineRow],
    op: &RowQuery,
    name: &str,
    need: Need,
    rank: impl Fn(bool) -> Option<u64>,
) -> Result<Chosen<RowInstantiation>, Refused> {
    let chosen = choose(
        name,
        rows.iter()
            .filter(|(entry, _)| entry.query == *op)
            .filter_map(|(entry, sharded)| {
                let launches = row_launch(entry);
                let candidate = Candidate::new(entry, &entry.symbol, launches, entry.invariance);
                Some(candidate.rank(rank(*sharded)?))
            }),
        need,
    )?;
    Ok(Chosen {
        entry: chosen.entry.clone(),
        guarantees: chosen.guarantees,
    })
}

/// What a row engine entry launches: its symbol, code (ABI and body),
/// grid and threads, under its query.
pub(super) fn row_launch(entry: &RowInstantiation) -> String {
    let RowInstantiation {
        profile: _,    // filters catalogs; no launch reads it
        symbol,        // the launched symbol
        query,         // selects the op
        width: _,      // compile-key metadata; the body, grid and arguments carry the widths
        grid,          // the plan
        threads,       // the plan
        params,        // the ABI
        body,          // the code
        invariance: _, // claims
    } = entry;
    format!("{symbol} {query:?} {grid:?} {threads} {params:?} {body}")
}

/// What a hyper-connection, PLE or n-gram entry launches: its symbol,
/// code, and the grouping and item layout its plans follow.
fn hyper_launch(entry: &Entry) -> String {
    let Entry {
        profile: _,    // filters catalogs; no launch reads it
        query,         // selects the op
        symbol,        // the launched symbol
        params,        // the ABI
        body,          // the code
        groups,        // the plan; the query's contract fixes it
        items,         // the plan; the query's contract fixes it
        invariance: _, // claims
    } = entry;
    format!("{symbol} {query:?} {params:?} {body} {groups} {items}")
}

/// What a grouped norm entry launches: its symbol, and its norm's
/// reduction, weights and groups, which its plan and code follow.
fn norm_launch(norm: &GroupedRmsNorm, symbol: &str) -> String {
    let GroupedRmsNorm {
        reduction,    // the code
        weight_dtype, // selects the op
        group_width,  // selects the op
        groups,       // selects the op
    } = norm;
    format!("{symbol} {reduction:?} {weight_dtype:?} {group_width} {groups}")
}

fn no_entry(op: &str) -> Refused {
    Refused::Rule(format!("no SM120 {op} entry serves the query"))
}

/// The embedding gather of `op` from a table starting at `row_offset`:
/// the model's whole-table entry when the offset is zero and it has one,
/// otherwise its sharded entry. The offset is the op's use: a sharded
/// entry subtracts it.
pub(super) fn embedding(
    selection: &Selection,
    op: &RowQuery,
    row_offset: u32,
    need: Need,
) -> Result<Prepared<EmbeddingBuffers<View>>, Refused> {
    let rows = &selection.engine;
    op.validate()?;
    let RowQuery::EmbeddingGather { width, .. } = *op else {
        return Err(Refused::Rule("expected an embedding gather".into()));
    };
    let entry = engine_entry(rows, op, "embedding", need, |sharded| {
        match (row_offset, sharded) {
            (0, false) => Some(0),
            (0, true) => Some(1),
            (_, sharded) => sharded.then_some(0),
        }
    })?;
    Ok(Box::new(Embedding {
        op: EngineOp::new(entry, need)?,
        width: u64::from(width),
        row_offset,
    }))
}

struct Embedding {
    op: EngineOp,
    width: u64,
    row_offset: u32,
}

impl PreparedOp<EmbeddingBuffers<View>> for Embedding {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &EmbeddingBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        // A step may grid a padded bucket over the rows its buffers hold:
        // rows past the device live count are not written, so every grid
        // that holds them writes equal bits.
        let grid = self.op.rows(Counts::rows(rows.max(counts.padded_rows)))?;
        if buffers.table_rows == 0 {
            return Err(Refusal("an embedding table holds at least one row"));
        }
        buffers
            .table
            .covers(dense(buffers.table_rows, self.width, 2)?)?;
        buffers.ids.covers(dense(rows, 1, 4)?)?;
        buffers.output.covers(dense(rows, self.width, 2)?)?;
        buffers.live_rows.covers(4)?;
        self.op.launch(
            grid,
            vec![
                buffer(buffers.table),
                buffer(buffers.ids),
                buffer(buffers.output),
                buffer(buffers.live_rows),
                Argument::U32(self.row_offset),
                Argument::U32(buffers.table_rows),
            ],
        )
    }
}

/// The argmax or top-k of `op` through the model's entry for it.
pub(super) fn logits(
    selection: &Selection,
    op: &RowQuery,
    need: Need,
) -> Result<Prepared<LogitsBuffers<View>>, Refused> {
    let rows = &selection.engine;
    op.validate()?;
    let (vocab, k) = match *op {
        RowQuery::LogitsArgmax { vocab } => (vocab, 1),
        RowQuery::LogitsTopK { vocab, k } => (vocab, k),
        _ => return Err(Refused::Rule("expected a logits argmax or top-k".into())),
    };
    let entry = engine_entry(rows, op, "logits", need, |_| Some(0))?;
    Ok(Box::new(Logits {
        op: EngineOp::new(entry, need)?,
        vocab: u64::from(vocab),
        k: u64::from(k),
    }))
}

struct Logits {
    op: EngineOp,
    vocab: u64,
    k: u64,
}

impl PreparedOp<LogitsBuffers<View>> for Logits {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &LogitsBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        buffers.logits.covers(dense(rows, self.vocab, 4)?)?;
        buffers.live_width.covers(4)?;
        buffers.ids.covers(dense(rows, self.k, 4)?)?;
        if let Some(log_probs) = buffers.log_probs {
            log_probs.covers(dense(rows, self.k, 4)?)?;
        }
        buffers.live_rows.covers(4)?;
        self.op.launch(
            rows,
            vec![
                buffer(buffers.logits),
                buffer(buffers.ids),
                buffers
                    .log_probs
                    .map_or(Argument::NullDevicePointer, buffer),
                buffer(buffers.live_rows),
                buffer(buffers.live_width),
            ],
        )
    }
}

/// Rank `rank`'s sharded argmax partial through the model's engine entry
/// of the query or a shared argmax entry of its logits.
pub(super) fn argmax_partial(
    selection: &Selection,
    op: &RowQuery,
    rank: u32,
    need: Need,
) -> Result<Prepared<ArgmaxPartialBuffers<View>>, Refused> {
    op.validate()?;
    let RowQuery::ShardedArgmaxPartial {
        vocab_per_rank,
        ranks,
        logits,
        ..
    } = *op
    else {
        return Err(Refused::Rule("expected a sharded argmax partial".into()));
    };
    if rank >= ranks {
        return Err(Refused::Rule(
            "the rank is past the sharded argmax's ranks".into(),
        ));
    }
    if u64::from(ranks) * u64::from(vocab_per_rank) > 1 << 31 {
        return Err(Refused::Rule(
            "the sharded argmax's vocabulary ids exceed I32".into(),
        ));
    }
    let vocab_offset = rank * vocab_per_rank;
    let logit_bytes = logit_bytes(logits)?;
    let pattern = argmax::Pattern::partial(logits);
    match argmax_entry(selection, op, pattern, "sharded argmax partial", need)? {
        Chosen {
            entry: ArgmaxEntry::Engine(entry),
            guarantees,
        } => Ok(Box::new(ArgmaxPartial {
            op: EngineOp::new(
                Chosen {
                    entry: *entry,
                    guarantees,
                },
                need,
            )?,
            vocab: u64::from(vocab_per_rank),
            logit_bytes,
            vocab_offset,
        })),
        Chosen {
            entry: ArgmaxEntry::Shared(entry),
            guarantees,
        } => Ok(Box::new(SharedPartial {
            op: SharedOp::new(entry, guarantees, need)?,
            vocab: vocab_per_rank,
            logit_bytes,
            vocab_offset,
        })),
    }
}

/// A sharded argmax entry that serves an op: a model's row engine entry
/// of the op's query, or a shared argmax entry of the op's pattern.
#[derive(Clone, Debug)]
enum ArgmaxEntry {
    Engine(Box<RowInstantiation>),
    Shared(argmax::Entry),
}

/// The entry that serves `op`: the engine entries whose query is `op` and
/// the shared entries of `pattern`.
fn argmax_entry(
    selection: &Selection,
    op: &RowQuery,
    pattern: Option<argmax::Pattern>,
    name: &str,
    need: Need,
) -> Result<Chosen<ArgmaxEntry>, Refused> {
    let engine = selection
        .engine
        .iter()
        .filter(|(entry, _)| entry.query == *op)
        .map(|(entry, _)| {
            Candidate::new(
                ArgmaxEntry::Engine(Box::new(entry.clone())),
                &entry.symbol,
                row_launch(entry),
                entry.invariance,
            )
        });
    let shared = selection
        .argmax
        .iter()
        .filter(|entry| Some(entry.pattern) == pattern)
        .map(|entry| {
            Candidate::new(
                ArgmaxEntry::Shared(entry.clone()),
                &entry.symbol,
                format!("{} {:?}", entry.symbol, entry.pattern),
                entry.guarantees(),
            )
        });
    choose(name, engine.chain(shared), need)
}

/// Bytes of one sharded argmax logit.
fn logit_bytes(logits: DType) -> Result<u64, Refused> {
    match logits {
        DType::Bf16 => Ok(2),
        DType::F32 => Ok(4),
        _ => Err(Refused::Rule(
            "sharded argmax logits are BF16 or F32".into(),
        )),
    }
}

struct ArgmaxPartial {
    op: EngineOp,
    vocab: u64,
    logit_bytes: u64,
    vocab_offset: u32,
}

impl PreparedOp<ArgmaxPartialBuffers<View>> for ArgmaxPartial {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &ArgmaxPartialBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        buffers
            .logits
            .covers(dense(rows, self.vocab, self.logit_bytes)?)?;
        buffers.live_width.covers(4)?;
        let pairs = buffers.pairs;
        pairs.scores.covers(dense(rows, 1, 4)?)?;
        pairs.ids.covers(dense(rows, 1, 4)?)?;
        buffers.live_rows.covers(4)?;
        let launches = self.op.launch(
            rows,
            vec![
                buffer(buffers.logits),
                buffer(pairs.scores),
                buffer(pairs.ids),
                buffer(buffers.live_rows),
                buffer(buffers.live_width),
                Argument::U32(self.vocab_offset),
            ],
        )?;
        peer_pairs(launches, pairs, [1, 2])
    }
}

/// The sharded argmax final through the model's engine entry of the
/// query, which reads row-major pairs, or a shared final entry.
pub(super) fn argmax_final(
    selection: &Selection,
    op: &RowQuery,
    need: Need,
) -> Result<Prepared<ArgmaxFinalBuffers<View>>, Refused> {
    op.validate()?;
    let RowQuery::ShardedArgmaxFinal { ranks, pairs, .. } = *op else {
        return Err(Refused::Rule("expected a sharded argmax final".into()));
    };
    if let ArgmaxLayout::RankMajor { stride } = pairs
        && need.capacity > stride
    {
        return Err(Refused::Rule(
            "a rank-major sharded argmax final's rows exceed its stride".into(),
        ));
    }
    let pattern = Some(argmax::Pattern::Final);
    match argmax_entry(selection, op, pattern, "sharded argmax final", need)? {
        Chosen {
            entry: ArgmaxEntry::Engine(entry),
            guarantees,
        } => {
            if pairs != ArgmaxLayout::RowMajor {
                return Err(Refused::Rule(
                    "a model's sharded argmax final entry reads row-major pairs".into(),
                ));
            }
            Ok(Box::new(ArgmaxFinal {
                op: EngineOp::new(
                    Chosen {
                        entry: *entry,
                        guarantees,
                    },
                    need,
                )?,
                ranks: u64::from(ranks),
            }))
        }
        Chosen {
            entry: ArgmaxEntry::Shared(entry),
            guarantees,
        } => Ok(Box::new(SharedFinal {
            op: SharedOp::new(entry, guarantees, need)?,
            ranks,
            pairs,
        })),
    }
}

struct ArgmaxFinal {
    op: EngineOp,
    ranks: u64,
}

impl PreparedOp<ArgmaxFinalBuffers<View>> for ArgmaxFinal {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &ArgmaxFinalBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        let pairs = buffers.pairs;
        let (Some(output), []) = (buffers.output, buffers.scatter.as_slice()) else {
            return Err(Refusal(
                "a model's sharded argmax final entry writes a dense output only",
            ));
        };
        pairs.scores.covers(dense(rows, self.ranks, 4)?)?;
        pairs.ids.covers(dense(rows, self.ranks, 4)?)?;
        output.covers(dense(rows, 1, 4)?)?;
        buffers.live_rows.covers(4)?;
        let launches = self.op.launch(
            rows,
            vec![
                buffer(pairs.scores),
                buffer(pairs.ids),
                buffer(output),
                buffer(buffers.live_rows),
            ],
        )?;
        peer_pairs(launches, pairs, [0, 1])
    }
}

/// One shared argmax entry launched at one CTA per row, up to the
/// prepared capacity, which may reach the backend's (`MAX_CAPACITY`): the
/// entries name no row bucket, so a launch plans its own rows.
struct SharedOp {
    entry: argmax::Entry,
    capacity: u32,
    info: OpInfo,
}

impl SharedOp {
    /// `entry`, claiming `guarantees`.
    fn new(entry: argmax::Entry, guarantees: Invariance, need: Need) -> Result<Self, Refused> {
        served(need)?;
        check_plan(&entry.plan(need.capacity)?)?;
        let implementation = match entry.pattern {
            argmax::Pattern::PartialF32 => "sm120.row.argmax.partial-f32.v1",
            argmax::Pattern::PartialBf16 => "sm120.row.argmax.partial-bf16.v1",
            argmax::Pattern::Final => "sm120.row.argmax.final.v1",
        };
        let info = OpInfo::new(implementation, guarantees, vec![(entry.symbol.clone(), 0)]);
        need.check(&info)?;
        Ok(Self {
            entry,
            capacity: need.capacity,
            info,
        })
    }

    /// The launch's rows: one to the prepared capacity.
    fn rows(&self, counts: Counts) -> Result<u32, Refusal> {
        if counts.rows == 0 || counts.rows > self.capacity {
            return Err(Refusal("row count outside the prepared capacity"));
        }
        Ok(counts.rows)
    }

    fn launch(&self, rows: u32, arguments: ArgumentList) -> Result<Vec<Launch>, Refusal> {
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows,
            plan: self.entry.plan(rows)?,
            arguments,
            collective: false,
            persistent: false,
        }])
    }
}

/// A rank's partial through a shared argmax entry.
struct SharedPartial {
    op: SharedOp,
    vocab: u32,
    logit_bytes: u64,
    vocab_offset: u32,
}

impl PreparedOp<ArgmaxPartialBuffers<View>> for SharedPartial {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &ArgmaxPartialBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        buffers
            .logits
            .covers(dense(rows, u64::from(self.vocab), self.logit_bytes)?)?;
        buffers.live_width.covers(4)?;
        let pairs = buffers.pairs;
        pairs.scores.covers(dense(rows, 1, 4)?)?;
        pairs.ids.covers(dense(rows, 1, 4)?)?;
        buffers.live_rows.covers(4)?;
        let arguments = argmax::partial_arguments(
            [
                buffers.logits,
                pairs.scores,
                pairs.ids,
                buffers.live_rows,
                buffers.live_width,
            ]
            .map(buffer),
            self.vocab,
            self.vocab_offset,
        )?;
        peer_pairs(self.op.launch(rows, arguments)?, pairs, [1, 2])
    }
}

/// The final through a shared argmax entry, over pairs of either layout.
struct SharedFinal {
    op: SharedOp,
    ranks: u32,
    pairs: ArgmaxLayout,
}

impl PreparedOp<ArgmaxFinalBuffers<View>> for SharedFinal {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &ArgmaxFinalBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        if buffers.scatter.len() > argmax::MAX_SCATTER {
            return Err(Refusal(
                "an SM120 sharded argmax final writes at most two scattered outputs",
            ));
        }
        if buffers.output.is_none() && buffers.scatter.is_empty() {
            return Err(Refusal("a sharded argmax final writes no output"));
        }
        distinct_outputs(buffers)?;
        let ranks = u64::from(self.ranks);
        let (row_stride, rank_stride, pairs) = match self.pairs {
            ArgmaxLayout::RowMajor => (self.ranks, 1, dense(rows, ranks, 4)?),
            ArgmaxLayout::RankMajor { stride } => (
                1,
                stride,
                (ranks - 1)
                    .checked_mul(u64::from(stride))
                    .and_then(|before| before.checked_add(u64::from(rows)))
                    .and_then(|elements| elements.checked_mul(4))
                    .ok_or(Refusal("row buffer bytes overflow"))?,
            ),
        };
        buffers.pairs.scores.covers(pairs)?;
        buffers.pairs.ids.covers(pairs)?;
        if let Some(output) = buffers.output {
            output.covers(dense(rows, 1, 4)?)?;
        }
        let none = (Argument::NullDevicePointer, Argument::NullDevicePointer, 0);
        let mut scatter = [none.clone(), none];
        for (to, target) in scatter.iter_mut().zip(&buffers.scatter) {
            target.rows.covers(dense(rows, 1, 4)?)?;
            // The entry compares an index read unsigned against the
            // elements, so at most 2^31 of them: every nonnegative I32
            // index, and no negative one, which reads as 2^31 or more.
            let elements = (target.output.bytes / 4).min(SCATTER_ELEMENTS) as u32;
            *to = (buffer(target.rows), buffer(target.output), elements);
        }
        buffers.live_rows.covers(4)?;
        let arguments = argmax::final_arguments(argmax::FinalArguments {
            scores: buffer(buffers.pairs.scores),
            ids: buffer(buffers.pairs.ids),
            output: buffers.output.map_or(Argument::NullDevicePointer, buffer),
            scatter,
            live_rows: buffer(buffers.live_rows),
            ranks: self.ranks,
            row_stride,
            rank_stride,
        })?;
        peer_pairs(self.op.launch(rows, arguments)?, buffers.pairs, [0, 1])
    }
}

/// The most elements of a scattered output a final's entry addresses: past
/// the largest I32 index, an index's unsigned reading would admit negative
/// ones.
const SCATTER_ELEMENTS: u64 = 1 << 31;

/// Refuses a final whose outputs share a view with each other or with an
/// input: the views the host can tell apart. Distinct destinations inside
/// one scattered output are device data, the contract's precondition.
fn distinct_outputs(buffers: &ArgmaxFinalBuffers<View>) -> Result<(), Refusal> {
    let outputs = buffers
        .output
        .iter()
        .chain(buffers.scatter.iter().map(|target| &target.output))
        .map(|view| view.handle)
        .collect::<Vec<_>>();
    let inputs = [buffers.pairs.scores, buffers.pairs.ids, buffers.live_rows]
        .into_iter()
        .chain(buffers.scatter.iter().map(|target| target.rows))
        .map(|view| view.handle)
        .collect::<Vec<_>>();
    for (at, output) in outputs.iter().enumerate() {
        if outputs[..at].contains(output) || inputs.contains(output) {
            return Err(Refusal(
                "a sharded argmax final's outputs share a view with each other or an input",
            ));
        }
    }
    Ok(())
}

/// Peer-backed pairs reach the entry as one-address pointer arrays: the
/// same parameter bytes as a pointer, but the host validates a peer lease
/// for them, where a plain buffer must belong to the launching rank.
fn peer_pairs(
    mut launches: Vec<Launch>,
    pairs: ArgmaxPairs<View>,
    at: [usize; 2],
) -> Result<Vec<Launch>, Refusal> {
    if pairs.peer {
        for launch in &mut launches {
            for index in at {
                let Some(Argument::DeviceBuffer(handle)) = launch.arguments.0.get(index) else {
                    return Err(Refusal("argmax pair arguments are not device buffers"));
                };
                launch.arguments.0[index] = Argument::DevicePointerArray(vec![*handle]);
            }
        }
    }
    Ok(launches)
}

/// One hyper-connection row entry, planned at the step's rows as its
/// builder plans it, up to the capacity its query was compiled for.
struct HyperOp {
    entry: Entry,
    /// The plan of each row count from one to the prepared capacity.
    plans: Vec<LaunchPlan>,
    info: OpInfo,
}

impl HyperOp {
    /// `chosen`'s entry, claiming what its registrations declare.
    fn new(chosen: Chosen<Entry>, need: Need) -> Result<Self, Refused> {
        let Chosen { entry, guarantees } = chosen;
        served(need)?;
        let plans = plans(need.capacity, |rows| entry.plan(rows, rows))?;
        let info = OpInfo::new(
            implementation(&entry.symbol, format!("{entry:?}")),
            guarantees,
            vec![(entry.symbol.clone(), 0)],
        );
        need.check(&info)?;
        Ok(Self { entry, plans, info })
    }

    fn rows(&self, counts: Counts) -> Result<u32, Refusal> {
        planned(&self.plans, counts).map(|_| counts.rows)
    }

    fn launch(&self, rows: u32, arguments: Vec<Argument>) -> Result<Vec<Launch>, Refusal> {
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows,
            plan: planned(&self.plans, Counts::rows(rows))?,
            arguments: self.entry.pack_arguments(arguments)?,
            collective: false,
            persistent: false,
        }])
    }
}

/// Whether `entry` reads a device live row count: the per-sequence form of
/// a row, not one over every decode item.
fn per_sequence(entry: &Entry) -> bool {
    entry.params.iter().any(|param| {
        matches!(param, kernel_build::EntryParam::DevicePointer { name, .. } if name == "live_rows")
    })
}

/// The model's per-sequence entry for `query`.
fn hyper_entry(
    entries: &[Entry],
    query: StaticQuery,
    op: &str,
    need: Need,
) -> Result<Chosen<Entry>, Refused> {
    let chosen = choose(
        op,
        entries
            .iter()
            .filter(|entry| entry.query == query && per_sequence(entry))
            .map(|entry| {
                let launches = hyper_launch(entry);
                Candidate::new(entry, &entry.symbol, launches, entry.invariance)
            }),
        need,
    )?;
    Ok(Chosen {
        entry: chosen.entry.clone(),
        guarantees: chosen.guarantees,
    })
}

/// One hyper-connection row of kind `K` and its row sizes, in elements.
struct Hyper<K> {
    op: HyperOp,
    /// `streams * width`, `width`, `streams` and the bottleneck rank, as
    /// the kind uses them.
    wide: u64,
    narrow: u64,
    kind: std::marker::PhantomData<K>,
}

fn hyper<K>(entries: &[Entry], op: H, need: Need, name: &str) -> Result<Box<Hyper<K>>, Refused> {
    op.validate()?;
    let (wide, narrow) = match op {
        H::GroupedNorm { streams, width, .. }
        | H::Mix { streams, width, .. }
        | H::Inject { streams, width, .. } => {
            (u64::from(streams) * u64::from(width), u64::from(width))
        }
        H::Bottleneck { rank, .. } => (u64::from(rank), u64::from(rank)),
        H::AllReduceInjectNorm { .. } => {
            return Err(Refused::Rule(
                "the hyper-connection all-reduce is a collective".into(),
            ));
        }
    };
    let entry = hyper_entry(entries, StaticQuery::HyperConnection(op), name, need)?;
    Ok(Box::new(Hyper {
        op: HyperOp::new(entry, need)?,
        wide,
        narrow,
        kind: std::marker::PhantomData,
    }))
}

/// The grouped norm of `op` over `weight` weights: the model's BF16
/// hyper-connection row for BF16 weights when it has one, otherwise its
/// grouped norm entry over weights of that dtype. Grouped norm entries of
/// different reductions or symbols are a tie: the op does not choose the
/// reduction order.
pub(super) fn grouped_norm(
    selection: &Selection,
    op: &H,
    weight: DType,
    need: Need,
) -> Result<Prepared<GroupedNormBuffers<View>>, Refused> {
    served(need)?;
    op.validate()?;
    let H::GroupedNorm {
        streams,
        width,
        eps_bits,
        capacity,
    } = *op
    else {
        return Err(Refused::Rule(
            "expected a hyper-connection grouped norm".into(),
        ));
    };
    let weight_bytes = match weight {
        DType::Bf16 => 2,
        DType::F32 => 4,
        _ => {
            return Err(Refused::Rule("grouped norm weights are BF16 or F32".into()));
        }
    };
    let query = StaticQuery::HyperConnection(*op);
    if weight == DType::Bf16 && selection.hyper.iter().any(|entry| entry.query == query) {
        return Ok(hyper::<GroupedNormBuffers<View>>(
            &selection.hyper,
            *op,
            need,
            "grouped norm",
        )?);
    }
    let (norm, symbol) = choose(
        "grouped norm",
        selection
            .grouped_norms
            .iter()
            .filter(|(norm, _)| {
                norm.weight_dtype == weight && norm.groups == streams && norm.group_width == width
            })
            // The entries declare no invariance.
            .map(|selected| {
                let (norm, symbol) = selected;
                Candidate::new(
                    selected,
                    symbol,
                    norm_launch(norm, symbol),
                    Invariance::NONE,
                )
            }),
        need,
    )?
    .entry;
    if eps_bits != HC_INJECT_NORM_EPS_BITS {
        return Err(Refused::Rule(
            "the SM120 grouped norm entries use epsilon 1e-6".into(),
        ));
    }
    if need.capacity > capacity {
        return Err(Refused::Rule(
            "the grouped norm capacity is outside its query's".into(),
        ));
    }
    norm.entry(symbol)?;
    let plans = plans(need.capacity, |rows| norm.plan(rows, rows))?;
    // The entries declare no invariance.
    let info = OpInfo::new(
        implementation(symbol, format!("{norm:?}")),
        Invariance::NONE,
        vec![(symbol.clone(), 0)],
    );
    need.check(&info)?;
    let wide = u64::from(streams) * u64::from(width);
    Ok(Box::new(GroupedNormOp {
        symbol: symbol.clone(),
        wide,
        weight_bytes: wide * weight_bytes,
        plans,
        info,
    }))
}

/// A grouped norm entry (`comm::epilogue::GroupedRmsNorm`), planned at the
/// step's rows as its builder plans it.
struct GroupedNormOp {
    symbol: String,
    /// `streams * width`.
    wide: u64,
    weight_bytes: u64,
    /// The plan of each row count from one to the prepared capacity.
    plans: Vec<LaunchPlan>,
    info: OpInfo,
}

impl PreparedOp<GroupedNormBuffers<View>> for GroupedNormOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &GroupedNormBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let plan = planned(&self.plans, counts)?;
        let rows = counts.rows;
        buffers.x.covers(dense(rows, self.wide, 2)?)?;
        buffers.weight.covers(self.weight_bytes)?;
        buffers.output.covers(dense(rows, self.wide, 2)?)?;
        buffers.live_rows.covers(4)?;
        let arguments = ArgumentList(vec![
            buffer(buffers.x),
            buffer(buffers.weight),
            buffer(buffers.output),
            buffer(buffers.live_rows),
        ]);
        arguments.validate()?;
        Ok(vec![Launch {
            symbol: self.symbol.clone(),
            rows,
            plan,
            arguments,
            collective: false,
            persistent: false,
        }])
    }
}

impl PreparedOp<GroupedNormBuffers<View>> for Hyper<GroupedNormBuffers<View>> {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &GroupedNormBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        buffers.x.covers(dense(rows, self.wide, 2)?)?;
        buffers.weight.covers(dense(1, self.wide, 2)?)?;
        buffers.output.covers(dense(rows, self.wide, 2)?)?;
        buffers.live_rows.covers(4)?;
        self.op.launch(
            rows,
            vec![
                buffer(buffers.x),
                buffer(buffers.weight),
                buffer(buffers.output),
                buffer(buffers.live_rows),
            ],
        )
    }
}

pub(super) fn bottleneck(
    selection: &Selection,
    op: &H,
    need: Need,
) -> Result<Prepared<BottleneckBuffers<View>>, Refused> {
    let entries = &selection.hyper;
    let H::Bottleneck { .. } = op else {
        return Err(Refused::Rule(
            "expected a hyper-connection bottleneck".into(),
        ));
    };
    Ok(hyper::<BottleneckBuffers<View>>(
        entries,
        *op,
        need,
        "bottleneck",
    )?)
}

impl PreparedOp<BottleneckBuffers<View>> for Hyper<BottleneckBuffers<View>> {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &BottleneckBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        buffers.down.covers(dense(rows, self.wide, 2)?)?;
        buffers.output.covers(dense(rows, self.wide, 2)?)?;
        buffers.live_rows.covers(4)?;
        self.op.launch(
            rows,
            vec![
                buffer(buffers.down),
                buffer(buffers.output),
                buffer(buffers.live_rows),
            ],
        )
    }
}

pub(super) fn mix(
    selection: &Selection,
    op: &H,
    need: Need,
) -> Result<Prepared<MixBuffers<View>>, Refused> {
    let entries = &selection.hyper;
    let H::Mix { .. } = op else {
        return Err(Refused::Rule("expected a hyper-connection mix".into()));
    };
    Ok(hyper::<MixBuffers<View>>(entries, *op, need, "mix")?)
}

impl PreparedOp<MixBuffers<View>> for Hyper<MixBuffers<View>> {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &MixBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        buffers.up.covers(dense(rows, self.wide, 2)?)?;
        buffers.normed.covers(dense(rows, self.wide, 2)?)?;
        buffers.output.covers(dense(rows, self.narrow, 2)?)?;
        buffers.live_rows.covers(4)?;
        self.op.launch(
            rows,
            vec![
                buffer(buffers.up),
                buffer(buffers.normed),
                buffer(buffers.output),
                buffer(buffers.live_rows),
            ],
        )
    }
}

pub(super) fn inject(
    selection: &Selection,
    op: &H,
    need: Need,
) -> Result<Prepared<InjectBuffers<View>>, Refused> {
    let entries = &selection.hyper;
    let H::Inject { .. } = op else {
        return Err(Refused::Rule("expected a hyper-connection inject".into()));
    };
    Ok(hyper::<InjectBuffers<View>>(entries, *op, need, "inject")?)
}

impl PreparedOp<InjectBuffers<View>> for Hyper<InjectBuffers<View>> {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &InjectBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        let streams = self.wide / self.narrow;
        buffers.x.covers(dense(rows, self.wide, 2)?)?;
        buffers.y.covers(dense(rows, self.narrow, 2)?)?;
        buffers.inject.covers(dense(rows, streams, 2)?)?;
        buffers.output.covers(dense(rows, self.wide, 2)?)?;
        buffers.live_rows.covers(4)?;
        self.op.launch(
            rows,
            vec![
                buffer(buffers.x),
                buffer(buffers.y),
                buffer(buffers.inject),
                buffer(buffers.output),
                buffer(buffers.live_rows),
            ],
        )
    }
}

/// Rank `rank`'s n-gram hash gather over one sequence chunk through the
/// model's per-sequence entry.
pub(super) fn ngram(
    selection: &Selection,
    op: &NgramQuery,
    eos: u32,
    rank: u32,
    need: Need,
) -> Result<Prepared<NgramBuffers<View>>, Refused> {
    op.validate()?;
    let NgramQuery::HashGather {
        orders,
        heads_per_order,
        head_width,
        local_heads,
        ..
    } = *op
    else {
        return Err(Refused::Rule("expected an n-gram hash gather".into()));
    };
    let heads = (orders - 1) * heads_per_order;
    if rank >= heads / local_heads {
        return Err(Refused::Rule("the rank is past the n-gram heads".into()));
    }
    let query = StaticQuery::Ngram(*op);
    let served = selection
        .ngrams
        .iter()
        .filter(|(entry, _)| entry.query == query && per_sequence(entry))
        .collect::<Vec<_>>();
    if served.is_empty() {
        return Err(no_entry("n-gram hash gather"));
    }
    // An entry's sources compile its end-of-document token in.
    if served.iter().all(|&&(_, compiled)| compiled != eos) {
        return Err(Refused::Rule(
            "the n-gram entries compile in another end-of-document token".into(),
        ));
    }
    let chosen = choose(
        "n-gram hash gather",
        served
            .into_iter()
            .filter(|&&(_, compiled)| compiled == eos)
            .map(|selected| {
                let entry = &selected.0;
                let launches = hyper_launch(entry);
                Candidate::new(selected, &entry.symbol, launches, entry.invariance)
            }),
        need,
    )?;
    let chosen = Chosen {
        entry: chosen.entry.0.clone(),
        guarantees: chosen.guarantees,
    };
    Ok(Box::new(Ngram {
        op: HyperOp::new(chosen, need)?,
        orders: u64::from(orders),
        heads: u64::from(heads),
        local: u64::from(local_heads) * u64::from(head_width),
        local_heads: u64::from(local_heads),
        head_width: u64::from(head_width),
        rank,
    }))
}

struct Ngram {
    op: HyperOp,
    orders: u64,
    heads: u64,
    /// The output elements of one row: the local heads' rows.
    local: u64,
    local_heads: u64,
    head_width: u64,
    rank: u32,
}

impl PreparedOp<NgramBuffers<View>> for Ngram {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &NgramBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        let history = (self.orders - 1) * 4;
        buffers.tokens.covers(dense(rows, 1, 4)?)?;
        buffers.history.covers(history)?;
        buffers.multipliers.covers(self.orders * 8)?;
        buffers.sizes.covers(self.heads * 8)?;
        buffers.offsets.covers(self.heads * 8)?;
        buffers.table.covers(
            buffers
                .table_rows
                .checked_mul(self.head_width)
                .ok_or(Refusal("row buffer bytes overflow"))?,
        )?;
        buffers.scale.covers(2)?;
        buffers.output.covers(dense(rows, self.local, 2)?)?;
        buffers.row_ids.covers(dense(rows, self.local_heads, 8)?)?;
        buffers.history_out.covers(history)?;
        buffers.live_rows.covers(4)?;
        if buffers.history_out.handle == buffers.history.handle {
            return Err(Refusal("the n-gram history output may not alias its input"));
        }
        self.op.launch(
            rows,
            vec![
                buffer(buffers.tokens),
                buffer(buffers.history),
                buffer(buffers.multipliers),
                buffer(buffers.sizes),
                buffer(buffers.offsets),
                buffer(buffers.table),
                buffer(buffers.scale),
                buffer(buffers.output),
                buffer(buffers.row_ids),
                buffer(buffers.history_out),
                buffer(buffers.live_rows),
                Argument::U32(self.rank),
                Argument::U64(buffers.table_start),
                Argument::U64(buffers.table_rows),
            ],
        )
    }
}

/// The PLE gate of `op` through the model's entry.
pub(super) fn ple_gate(
    selection: &Selection,
    op: &NgramQuery,
    need: Need,
) -> Result<Prepared<PleGateBuffers<View>>, Refused> {
    let entries = &selection.hyper;
    op.validate()?;
    let NgramQuery::PleGate { streams, width, .. } = *op else {
        return Err(Refused::Rule("expected a PLE gate".into()));
    };
    let entry = hyper_entry(entries, StaticQuery::Ngram(*op), "PLE gate", need)?;
    Ok(Box::new(Hyper::<PleGateBuffers<View>> {
        op: HyperOp::new(entry, need)?,
        wide: u64::from(streams) * u64::from(width),
        narrow: u64::from(width),
        kind: std::marker::PhantomData,
    }))
}

impl PreparedOp<PleGateBuffers<View>> for Hyper<PleGateBuffers<View>> {
    fn info(&self) -> &OpInfo {
        &self.op.info
    }

    fn launches(
        &self,
        buffers: &PleGateBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = self.op.rows(counts)?;
        buffers.key.covers(dense(rows, self.wide, 2)?)?;
        buffers.query.covers(dense(rows, self.wide, 2)?)?;
        buffers.value.covers(dense(rows, self.narrow, 2)?)?;
        buffers.output.covers(dense(rows, self.wide, 2)?)?;
        buffers.live_rows.covers(4)?;
        self.op.launch(
            rows,
            vec![
                buffer(buffers.key),
                buffer(buffers.query),
                buffer(buffers.value),
                buffer(buffers.output),
                buffer(buffers.live_rows),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Profile;
    use crate::backend::select::orders;
    use crate::backend::{Sm120, view};
    use crate::row::argmax::FINAL_SYMBOL as FINAL;
    use crate::row::hyper_connection::{
        Ngram as NgramShape, Streams, hc_entries, ngram_decode_entry, ngram_entry, ple_gate_entry,
    };
    use crate::row::{Grid, embedding as embedding_entry, logits as logits_entry};
    use kernel_build::EntryParam;
    use kernel_core::backend::Backend;
    use kernel_core::contract::NormReduction;
    use kernel_core::properties::Chunk;
    use kernel_core::rows::ArgmaxScatter;

    const PROFILE: Profile = Profile::StdDenseMoe;
    const VOCAB: u32 = 152_576;
    const WIDTH: u32 = 4096;

    fn ptr(name: &str, ty: &str) -> EntryParam {
        EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        }
    }

    /// What a model's builder call gives for `entry` over `rows` rows at
    /// the smallest bucket holding them.
    fn built(entry: &RowInstantiation, rows: u32, arguments: Vec<Argument>) -> Launch {
        let bucket = smallest_bucket(rows).unwrap();
        Launch {
            symbol: entry.symbol.clone(),
            rows,
            plan: entry.plan(bucket, counts(rows)).unwrap(),
            arguments: entry.pack_arguments(arguments).unwrap(),
            collective: false,
            persistent: false,
        }
    }

    fn built_row(entry: &Entry, rows: u32, arguments: Vec<Argument>) -> Launch {
        Launch {
            symbol: entry.symbol.clone(),
            rows,
            plan: entry.plan(rows, rows).unwrap(),
            arguments: entry.pack_arguments(arguments).unwrap(),
            collective: false,
            persistent: false,
        }
    }

    fn handles(handles: &[u64]) -> Vec<Argument> {
        handles.iter().map(|&h| Argument::DeviceBuffer(h)).collect()
    }

    #[test]
    fn embedding_launches_what_the_builder_plans() {
        let query = RowQuery::EmbeddingGather {
            vocab: VOCAB,
            width: WIDTH,
        };
        let full = embedding_entry(PROFILE, "embedding_full", VOCAB, WIDTH, false);
        let shard = embedding_entry(PROFILE, "embedding_shard", VOCAB, WIDTH, true);
        let backend = Sm120::new()
            .with_sharded_embedding(shard.clone())
            .with_row(full.clone());
        let row = u64::from(WIDTH) * 2;
        // A whole table, its offset zero, through the whole-table entry.
        let prepared = backend.embedding_gather(&query, 0, Need::rows(32)).unwrap();
        assert_eq!(prepared.info().guarantees, full.invariance);
        assert_eq!(prepared.info().symbols, [(full.symbol.clone(), 0)]);
        let buffers = EmbeddingBuffers {
            table: view(1, u64::from(VOCAB) * row),
            ids: view(2, 32 * 4),
            output: view(3, 32 * row),
            live_rows: view(4, 4),
            table_rows: VOCAB,
        };
        for rows in [1, 3, 32] {
            let mut arguments = handles(&[1, 2, 3, 4]);
            arguments.extend([Argument::U32(0), Argument::U32(VOCAB)]);
            assert_eq!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                [built(&full, rows, arguments)]
            );
        }
        // A row gather from fewer rows than its capacity.
        let gather = EmbeddingBuffers {
            table: view(1, 40 * row),
            table_rows: 40,
            ..buffers
        };
        let mut arguments = handles(&[1, 2, 3, 4]);
        arguments.extend([Argument::U32(0), Argument::U32(40)]);
        assert_eq!(
            prepared
                .launches(&gather, View::EMPTY, Counts::rows(8))
                .unwrap(),
            [built(&full, 8, arguments)]
        );
        // One rank's shard through the sharded entry.
        let shard_rows = VOCAB / 4;
        let prepared = backend
            .embedding_gather(&query, 2 * shard_rows, Need::rows(4096))
            .unwrap();
        let buffers = EmbeddingBuffers {
            table: view(1, u64::from(shard_rows) * row),
            ids: view(2, 4096 * 4),
            output: view(3, 4096 * row),
            live_rows: view(4, 4),
            table_rows: shard_rows,
        };
        let mut arguments = handles(&[1, 2, 3, 4]);
        arguments.extend([Argument::U32(2 * shard_rows), Argument::U32(shard_rows)]);
        assert_eq!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(300))
                .unwrap(),
            [built(&shard, 300, arguments)]
        );
        // Sizes and counts.
        for short in [
            EmbeddingBuffers {
                table: view(1, u64::from(shard_rows) * row - 2),
                ..buffers
            },
            EmbeddingBuffers {
                ids: view(2, 300 * 4 - 4),
                ..buffers
            },
            EmbeddingBuffers {
                output: view(3, 300 * row - 2),
                ..buffers
            },
            EmbeddingBuffers {
                live_rows: view(4, 2),
                ..buffers
            },
            EmbeddingBuffers {
                table_rows: 0,
                ..buffers
            },
        ] {
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(300))
                    .is_err()
            );
        }
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(0))
                .is_err()
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(4097))
                .is_err()
        );
        // A shard offset needs an entry that subtracts it.
        assert!(matches!(
            Sm120::new()
                .with_row(full)
                .embedding_gather(&query, 1, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
        // With no whole-table entry, a zero offset takes the sharded one.
        let prepared = Sm120::new()
            .with_sharded_embedding(shard.clone())
            .embedding_gather(&query, 0, Need::rows(4))
            .unwrap();
        assert_eq!(prepared.info().symbols, [(shard.symbol, 0)]);
    }

    /// A step that grids a padded bucket while its ids and output hold
    /// fewer rows passes those rows as `Counts::rows` and the bucket as
    /// `Counts::padded_rows`: buffers are checked at the rows, and the
    /// launch grids and reports the bucket.
    #[test]
    fn padded_embedding_rows_grid_the_bucket_over_the_rows_the_buffers_hold() {
        let query = RowQuery::EmbeddingGather {
            vocab: VOCAB,
            width: WIDTH,
        };
        let full = embedding_entry(PROFILE, "embedding_full", VOCAB, WIDTH, false);
        let prepared = Sm120::new()
            .with_row(full)
            .embedding_gather(&query, 0, Need::rows(4096))
            .unwrap();
        let row = u64::from(WIDTH) * 2;
        let (rows, bucket) = (600_u32, 1024_u32);
        let buffers = |rows: u32| EmbeddingBuffers {
            table: view(1, u64::from(VOCAB) * row),
            ids: view(2, u64::from(rows) * 4),
            output: view(3, u64::from(rows) * row),
            live_rows: view(4, 4),
            table_rows: VOCAB,
        };
        let padded = |padded_rows| Counts {
            padded_rows,
            ..Counts::rows(rows)
        };
        assert_eq!(
            prepared
                .launches(&buffers(rows), View::EMPTY, padded(bucket))
                .unwrap(),
            prepared
                .launches(&buffers(bucket), View::EMPTY, Counts::rows(bucket))
                .unwrap()
        );
        // Padding below the rows grids the rows.
        assert_eq!(
            prepared
                .launches(&buffers(rows), View::EMPTY, padded(1))
                .unwrap(),
            prepared
                .launches(&buffers(rows), View::EMPTY, Counts::rows(rows))
                .unwrap()
        );
        assert!(
            prepared
                .launches(&buffers(rows), View::EMPTY, Counts::rows(bucket))
                .is_err()
        );
        assert!(
            prepared
                .launches(&buffers(rows), View::EMPTY, padded(8192))
                .is_err()
        );
    }

    #[test]
    fn logits_launch_what_the_builder_plans() {
        let shard = VOCAB / 4;
        let argmax = logits_entry(PROFILE, "argmax_full", VOCAB, 1);
        let top = logits_entry(PROFILE, "top20_shard", shard, 20);
        let backend = Sm120::new().with_row(argmax.clone()).with_row(top.clone());
        for (entry, query, vocab, k) in [
            (
                &argmax,
                RowQuery::LogitsArgmax { vocab: VOCAB },
                VOCAB,
                1_u32,
            ),
            (
                &top,
                RowQuery::LogitsTopK {
                    vocab: shard,
                    k: 20,
                },
                shard,
                20,
            ),
        ] {
            let prepared = backend.logits_top_k(&query, Need::rows(64)).unwrap();
            assert_eq!(prepared.info().guarantees, entry.invariance);
            let (rows, vocab, k) = (64, u64::from(vocab), u64::from(k));
            let buffers = LogitsBuffers {
                logits: view(1, rows * vocab * 4),
                live_width: view(2, 4),
                ids: view(3, rows * k * 4),
                log_probs: Some(view(4, rows * k * 4)),
                live_rows: view(5, 4),
            };
            for rows in [1, 17, 64] {
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    [built(entry, rows, handles(&[1, 3, 4, 5, 2]))]
                );
            }
            // Without log-probabilities the entry takes a null pointer.
            let ids_only = LogitsBuffers {
                log_probs: None,
                ..buffers
            };
            let mut arguments = handles(&[1, 3]);
            arguments.extend([Argument::NullDevicePointer]);
            arguments.extend(handles(&[5, 2]));
            assert_eq!(
                prepared
                    .launches(&ids_only, View::EMPTY, Counts::rows(2))
                    .unwrap(),
                [built(entry, 2, arguments)]
            );
            for short in [
                LogitsBuffers {
                    logits: view(1, rows * vocab * 4 - 4),
                    ..buffers
                },
                LogitsBuffers {
                    ids: view(3, rows * k * 4 - 4),
                    ..buffers
                },
                LogitsBuffers {
                    log_probs: Some(view(4, 4)),
                    ..buffers
                },
                LogitsBuffers {
                    live_width: view(2, 0),
                    ..buffers
                },
            ] {
                assert!(
                    prepared
                        .launches(&short, View::EMPTY, Counts::rows(64))
                        .is_err()
                );
            }
            assert!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(65))
                    .is_err()
            );
        }
        // An unserved variant, a capacity past every bucket, and exactness
        // the entries do not declare.
        assert!(matches!(
            backend.logits_top_k(&RowQuery::LogitsArgmax { vocab: shard }, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            backend.logits_top_k(
                &RowQuery::EmbeddingGather {
                    vocab: VOCAB,
                    width: WIDTH
                },
                Need::rows(4)
            ),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            backend.logits_top_k(&RowQuery::LogitsArgmax { vocab: VOCAB }, Need::rows(8193)),
            Err(Refused::Rule(rule)) if rule.contains("bucket")
        ));
        assert!(matches!(
            backend.logits_top_k(
                &RowQuery::LogitsArgmax { vocab: VOCAB },
                Need::rows(4).with_exact(Invariance::any_rows(4))
            ),
            Err(Refused::Inexact { .. })
        ));
    }

    /// A BF16 sharded argmax partial, as a model's own entries declare it.
    fn partial_query(vocab_per_rank: u32, ranks: u32) -> RowQuery {
        RowQuery::ShardedArgmaxPartial {
            vocab_per_rank,
            ranks,
            logits: DType::Bf16,
        }
    }

    /// The final over row-major pairs.
    fn final_query(ranks: u32) -> RowQuery {
        RowQuery::ShardedArgmaxFinal {
            ranks,
            pairs: ArgmaxLayout::RowMajor,
        }
    }

    /// The sharded argmax entries as a model's catalog builds them.
    fn argmax_entries(vocab: u32, ranks: u32) -> (RowInstantiation, RowInstantiation) {
        let partial = RowInstantiation::new(
            PROFILE,
            "test_argmax_partial",
            partial_query(vocab, ranks),
            vocab,
            (Grid::Rows, 256),
            vec![
                ptr("logits", "const uint16_t"),
                ptr("scores_out", "float"),
                ptr("ids_out", "int32_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("live_width", "const uint32_t"),
                EntryParam::U32 {
                    name: "vocab_offset".into(),
                },
            ],
            format!(
                "chop::row::dflash_argmax_partial<{vocab}>(logits,scores_out,ids_out,live_rows,live_width,vocab_offset);"
            ),
        );
        let last = RowInstantiation::new(
            PROFILE,
            "test_argmax_final",
            final_query(ranks),
            ranks,
            (Grid::Rows, 128),
            vec![
                ptr("scores", "const float"),
                ptr("ids", "const int32_t"),
                ptr("output", "int32_t"),
                ptr("live_rows", "const uint32_t"),
            ],
            format!("chop::row::dflash_argmax_final<{ranks}>(scores,ids,output,live_rows);"),
        );
        (partial, last)
    }

    #[test]
    fn sharded_argmax_launches_what_the_builder_plans() {
        let (vocab, ranks) = (VOCAB / 4, 4);
        let (partial, last) = argmax_entries(vocab, ranks);
        let backend = Sm120::new()
            .with_row(partial.clone())
            .with_row(last.clone());
        let query = partial_query(vocab, ranks);
        let prepared = backend
            .sharded_argmax_partial(&query, 3, Need::rows(8))
            .unwrap();
        // Pairs in the rank's own memory, as a speculative drafter's
        // program passes them.
        let buffers = ArgmaxPartialBuffers {
            logits: view(1, 8 * u64::from(vocab) * 2),
            live_width: view(2, 4),
            pairs: ArgmaxPairs {
                scores: view(3, 8 * 4),
                ids: view(4, 8 * 4),
                peer: false,
            },
            live_rows: view(5, 4),
        };
        for rows in [1, 5, 8] {
            let mut arguments = handles(&[1, 3, 4, 5, 2]);
            arguments.push(Argument::U32(3 * vocab));
            assert_eq!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                [built(&partial, rows, arguments)]
            );
        }
        // Pairs in the root rank's memory, written through peer leases.
        let peer = ArgmaxPartialBuffers {
            pairs: ArgmaxPairs {
                peer: true,
                ..buffers.pairs
            },
            ..buffers
        };
        for rows in [1, 3, 8] {
            assert_eq!(
                prepared
                    .launches(&peer, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                [adapted(
                    &partial,
                    rows,
                    &[1, 3, 4, 5, 2],
                    Some(3 * vocab),
                    [1, 2]
                )]
            );
        }
        let short = ArgmaxPartialBuffers {
            logits: view(1, 8 * u64::from(vocab) * 2 - 2),
            ..buffers
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(8))
                .is_err()
        );
        let short = ArgmaxPartialBuffers {
            pairs: ArgmaxPairs {
                ids: view(4, 8 * 4 - 4),
                ..peer.pairs
            },
            ..peer
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(8))
                .is_err()
        );
        assert!(matches!(
            backend.sharded_argmax_partial(&query, ranks, Need::rows(8)),
            Err(Refused::Rule(_))
        ));
        let query = final_query(ranks);
        let prepared = backend.sharded_argmax_final(&query, Need::rows(8)).unwrap();
        let buffers = ArgmaxFinalBuffers {
            pairs: ArgmaxPairs {
                scores: view(1, 8 * 4 * 4),
                ids: view(2, 8 * 4 * 4),
                peer: false,
            },
            output: Some(view(3, 8 * 4)),
            scatter: Vec::new(),
            live_rows: view(4, 4),
        };
        assert_eq!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(8))
                .unwrap(),
            [built(&last, 8, handles(&[1, 2, 3, 4]))]
        );
        let peer = ArgmaxFinalBuffers {
            pairs: ArgmaxPairs {
                peer: true,
                ..buffers.pairs
            },
            ..buffers.clone()
        };
        for rows in [1, 6, 8] {
            assert_eq!(
                prepared
                    .launches(&peer, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                [adapted(&last, rows, &[1, 2, 3, 4], None, [0, 1])]
            );
        }
        let short = ArgmaxFinalBuffers {
            pairs: ArgmaxPairs {
                ids: view(2, 8 * 4 * 4 - 4),
                ..buffers.pairs
            },
            ..buffers.clone()
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(8))
                .is_err()
        );
        // The entry's ABI writes one dense output.
        let scattered = ArgmaxFinalBuffers {
            scatter: vec![ArgmaxScatter {
                rows: view(5, 8 * 4),
                output: view(6, 64 * 4),
            }],
            ..buffers.clone()
        };
        assert!(
            prepared
                .launches(&scattered, View::EMPTY, Counts::rows(8))
                .is_err()
        );
        let undirected = ArgmaxFinalBuffers {
            output: None,
            ..buffers
        };
        assert!(
            prepared
                .launches(&undirected, View::EMPTY, Counts::rows(8))
                .is_err()
        );
        assert!(matches!(
            backend.sharded_argmax_final(&final_query(2), Need::rows(8)),
            Err(Refused::Rule(_))
        ));
        let rank_major = RowQuery::ShardedArgmaxFinal {
            ranks,
            pairs: ArgmaxLayout::RankMajor { stride: 8 },
        };
        assert!(matches!(
            backend.sharded_argmax_final(&rank_major, Need::rows(8)),
            Err(Refused::Rule(rule)) if rule.contains("no SM120")
        ));
        // A model's engine entry and a shared entry of one query tie.
        let both = Sm120::new()
            .with_row(last.clone())
            .with_row_argmax(argmax::Entry::new(
                argmax::Pattern::Final,
                argmax::FINAL_SYMBOL,
            ));
        assert!(matches!(
            both.sharded_argmax_final(&query, Need::rows(8)),
            Err(Refused::Rule(rule)) if rule.contains("two SM120")
        ));
        // An engine entry's ABI reads row-major pairs, whatever its query.
        let misdeclared = Sm120::new().with_row(RowInstantiation {
            query: rank_major,
            ..last.clone()
        });
        assert!(matches!(
            misdeclared.sharded_argmax_final(&rank_major, Need::rows(8)),
            Err(Refused::Rule(rule)) if rule.contains("row-major")
        ));
    }

    /// A shared sharded argmax launch of `rows` rows.
    fn shared(
        symbol: &str,
        pattern: argmax::Pattern,
        rows: u32,
        arguments: Vec<Argument>,
    ) -> Launch {
        Launch {
            symbol: symbol.into(),
            rows,
            plan: argmax::plan(pattern, rows).unwrap(),
            arguments: ArgumentList(arguments),
            collective: false,
            persistent: false,
        }
    }

    #[test]
    fn shared_argmax_serves_either_logit_and_layout() {
        let (vocab, ranks, stride) = (1000, 4, 16);
        let mut backend = Sm120::new();
        for entry in argmax::entries() {
            backend = backend.with_row_argmax(entry);
        }
        let query = |vocab_per_rank, logits| RowQuery::ShardedArgmaxPartial {
            vocab_per_rank,
            ranks,
            logits,
        };
        let partial = |logits| query(vocab, logits);
        for (logits, bytes, symbol, pattern) in [
            (
                DType::F32,
                4,
                argmax::PARTIAL_F32_SYMBOL,
                argmax::Pattern::PartialF32,
            ),
            (
                DType::Bf16,
                2,
                argmax::PARTIAL_BF16_SYMBOL,
                argmax::Pattern::PartialBf16,
            ),
        ] {
            let prepared = backend
                .sharded_argmax_partial(&partial(logits), 2, Need::rows(16))
                .unwrap();
            assert_eq!(prepared.info().guarantees, argmax::INVARIANCE);
            assert_eq!(prepared.info().symbols, [(symbol.to_owned(), 0)]);
            let buffers = ArgmaxPartialBuffers {
                logits: view(1, 16 * u64::from(vocab) * bytes),
                live_width: view(2, 4),
                pairs: ArgmaxPairs {
                    scores: view(3, 16 * 4),
                    ids: view(4, 16 * 4),
                    peer: false,
                },
                live_rows: view(5, 4),
            };
            for rows in [1, 7, 16] {
                let mut arguments = handles(&[1, 3, 4, 5, 2]);
                arguments.extend([Argument::U32(vocab), Argument::U32(2 * vocab)]);
                assert_eq!(
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap(),
                    [shared(symbol, pattern, rows, arguments)]
                );
            }
            let short = ArgmaxPartialBuffers {
                logits: view(1, 16 * u64::from(vocab) * bytes - 1),
                ..buffers
            };
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(16))
                    .is_err()
            );
            assert!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(17))
                    .is_err()
            );
        }
        let huge = query(1 << 30, DType::F32);
        assert!(matches!(
            backend.sharded_argmax_partial(&huge, 0, Need::rows(16)),
            Err(Refused::Rule(rule)) if rule.contains("I32")
        ));
        // The entries plan their own rows: a partial serves every row count
        // its claims name, past the row engine's largest bucket.
        let narrow = query(1, DType::F32);
        let max = crate::backend::MAX_CAPACITY;
        let prepared = backend
            .sharded_argmax_partial(&narrow, 1, Need::rows(max))
            .unwrap();
        assert!(prepared.info().guarantees.one_row_at(max));
        let wide = ArgmaxPartialBuffers {
            logits: view(1, u64::from(max) * 4),
            live_width: view(2, 4),
            pairs: ArgmaxPairs {
                scores: view(3, u64::from(max) * 4),
                ids: view(4, u64::from(max) * 4),
                peer: false,
            },
            live_rows: view(5, 4),
        };
        for rows in [8193, max] {
            let launched = prepared
                .launches(&wide, View::EMPTY, Counts::rows(rows))
                .unwrap();
            assert_eq!(launched[0].plan.grid, [rows, 1, 1]);
        }
        assert!(
            prepared
                .launches(&wide, View::EMPTY, Counts::rows(max + 1))
                .is_err()
        );
        assert!(
            backend
                .sharded_argmax_partial(&narrow, 1, Need::rows(max + 1))
                .is_err()
        );

        let last = |pairs| RowQuery::ShardedArgmaxFinal { ranks, pairs };
        let rank_major = last(ArgmaxLayout::RankMajor { stride });
        let prepared = backend
            .sharded_argmax_final(&rank_major, Need::rows(stride))
            .unwrap();
        assert_eq!(prepared.info().guarantees, argmax::FINAL_INVARIANCE);
        // Rows copied from one row share its destinations, so the final
        // claims no batch independence: a model that needs it is refused.
        assert!(matches!(
            backend.sharded_argmax_final(
                &last(ArgmaxLayout::RowMajor),
                Need::rows(2).with_exact(Invariance::any_rows(2))
            ),
            Err(Refused::Inexact { .. })
        ));
        assert!(
            backend
                .sharded_argmax_final(
                    &last(ArgmaxLayout::RowMajor),
                    Need::rows(2).with_exact(Invariance::one_row(1).with_chunk(Chunk::Any))
                )
                .is_ok()
        );
        // The last rank's pairs end `rows` past its stride's start.
        let gathered = (3 * u64::from(stride) + 5) * 4;
        let buffers = ArgmaxFinalBuffers {
            pairs: ArgmaxPairs {
                scores: view(1, gathered),
                ids: view(2, gathered),
                peer: false,
            },
            output: None,
            scatter: vec![
                ArgmaxScatter {
                    rows: view(3, 5 * 4),
                    output: view(4, 40 * 4),
                },
                ArgmaxScatter {
                    rows: view(5, 5 * 4),
                    output: view(6, 9 * 4 + 2),
                },
            ],
            live_rows: view(7, 4),
        };
        let mut arguments = handles(&[1, 2]);
        arguments.push(Argument::NullDevicePointer);
        arguments.extend(handles(&[3, 4, 5, 6, 7]));
        arguments.extend([4, 1, stride, 40, 9].map(Argument::U32));
        assert_eq!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(5))
                .unwrap(),
            [shared(FINAL, argmax::Pattern::Final, 5, arguments)]
        );
        // A scattered output past 2^31 elements is addressed as 2^31 of
        // them, so a negative index (2^31 or more unsigned) writes nothing.
        let huge = ArgmaxFinalBuffers {
            scatter: vec![
                ArgmaxScatter {
                    output: view(4, 4 * u64::from(u32::MAX)),
                    ..buffers.scatter[0]
                },
                ArgmaxScatter {
                    output: view(6, 4 << 40),
                    ..buffers.scatter[1]
                },
            ],
            ..buffers.clone()
        };
        let launched = prepared
            .launches(&huge, View::EMPTY, Counts::rows(5))
            .unwrap();
        assert_eq!(
            launched[0].arguments.0[11..],
            [Argument::U32(1 << 31), Argument::U32(1 << 31)]
        );
        // Peer pairs reach the entry as one-address arrays.
        let peer = ArgmaxFinalBuffers {
            pairs: ArgmaxPairs {
                peer: true,
                ..buffers.pairs
            },
            ..buffers.clone()
        };
        let launched = prepared
            .launches(&peer, View::EMPTY, Counts::rows(5))
            .unwrap();
        assert_eq!(
            launched[0].arguments.0[..2],
            [
                Argument::DevicePointerArray(vec![1]),
                Argument::DevicePointerArray(vec![2])
            ]
        );
        for wrong in [
            ArgmaxFinalBuffers {
                pairs: ArgmaxPairs {
                    ids: view(2, gathered - 4),
                    ..buffers.pairs
                },
                ..buffers.clone()
            },
            ArgmaxFinalBuffers {
                scatter: Vec::new(),
                ..buffers.clone()
            },
            ArgmaxFinalBuffers {
                scatter: vec![buffers.scatter[0]; 3],
                ..buffers.clone()
            },
            ArgmaxFinalBuffers {
                scatter: vec![ArgmaxScatter {
                    rows: view(3, 4 * 4),
                    ..buffers.scatter[0]
                }],
                ..buffers.clone()
            },
        ] {
            assert!(
                prepared
                    .launches(&wrong, View::EMPTY, Counts::rows(5))
                    .is_err()
            );
        }
        // Outputs sharing a view with each other or with an input.
        let scatter = |rows, output| ArgmaxScatter { rows, output };
        for shared in [
            ArgmaxFinalBuffers {
                output: Some(view(4, 5 * 4)),
                ..buffers.clone()
            },
            ArgmaxFinalBuffers {
                scatter: vec![buffers.scatter[0], scatter(view(5, 5 * 4), view(4, 40 * 4))],
                ..buffers.clone()
            },
            ArgmaxFinalBuffers {
                scatter: vec![buffers.scatter[0], scatter(view(5, 5 * 4), view(1, 40 * 4))],
                ..buffers.clone()
            },
            ArgmaxFinalBuffers {
                output: Some(view(3, 5 * 4)),
                ..buffers.clone()
            },
            ArgmaxFinalBuffers {
                output: Some(view(7, 5 * 4)),
                ..buffers.clone()
            },
        ] {
            assert!(matches!(
                prepared.launches(&shared, View::EMPTY, Counts::rows(5)),
                Err(Refusal(rule)) if rule.contains("share a view")
            ));
        }
        assert!(
            prepared
                .launches(
                    &ArgmaxFinalBuffers {
                        output: Some(view(8, 5 * 4)),
                        ..buffers.clone()
                    },
                    View::EMPTY,
                    Counts::rows(5)
                )
                .is_ok()
        );
        assert!(matches!(
            backend.sharded_argmax_final(&rank_major, Need::rows(stride + 1)),
            Err(Refused::Rule(rule)) if rule.contains("stride")
        ));
        // Row-major pairs into a dense output.
        let prepared = backend
            .sharded_argmax_final(&last(ArgmaxLayout::RowMajor), Need::rows(stride))
            .unwrap();
        let dense_buffers = ArgmaxFinalBuffers {
            pairs: ArgmaxPairs {
                scores: view(1, 5 * 4 * 4),
                ids: view(2, 5 * 4 * 4),
                peer: false,
            },
            output: Some(view(8, 5 * 4)),
            scatter: Vec::new(),
            live_rows: view(7, 4),
        };
        let mut arguments = handles(&[1, 2, 8]);
        arguments.extend(vec![Argument::NullDevicePointer; 4]);
        arguments.extend(handles(&[7]));
        arguments.extend([4, 4, 1, 0, 0].map(Argument::U32));
        assert_eq!(
            prepared
                .launches(&dense_buffers, View::EMPTY, Counts::rows(5))
                .unwrap(),
            [shared(FINAL, argmax::Pattern::Final, 5, arguments)]
        );
        // A model's own symbol of a pattern claims nothing, and two entries
        // of one pattern tie.
        let own = Sm120::new().with_row_argmax(argmax::Entry::new(
            argmax::Pattern::Final,
            "chop_row_test_argmax_final_v1",
        ));
        let prepared = own
            .sharded_argmax_final(&rank_major, Need::rows(stride))
            .unwrap();
        assert_eq!(prepared.info().guarantees, Invariance::NONE);
        assert!(
            own.sharded_argmax_final(
                &rank_major,
                Need::rows(stride).with_exact(Invariance::one_row(stride))
            )
            .is_err()
        );
        let both = backend.with_row_argmax(argmax::Entry::new(
            argmax::Pattern::Final,
            "chop_row_test_argmax_final_v1",
        ));
        assert!(matches!(
            both.sharded_argmax_final(&rank_major, Need::rows(stride)),
            Err(Refused::Rule(rule)) if rule.contains("two SM120")
        ));
    }

    /// What a model's sharded-argmax adapter builds today for pairs behind
    /// peer leases: the entry planned at `rows.next_power_of_two()`, then
    /// the pair pointers at `at` rewritten as one-address arrays.
    fn adapted(
        entry: &RowInstantiation,
        rows: u32,
        buffers: &[u64],
        offset: Option<u32>,
        at: [usize; 2],
    ) -> Launch {
        let mut arguments = handles(buffers);
        arguments.extend(offset.map(Argument::U32));
        let plan = entry
            .plan(
                rows.next_power_of_two(),
                RuntimeCounts {
                    m: rows,
                    batch: rows,
                    context: 1,
                    routed_rows: 1,
                    split_k: 1,
                },
            )
            .unwrap();
        let mut arguments = entry.pack_arguments(arguments).unwrap();
        for index in at {
            let Argument::DeviceBuffer(id) = arguments.0[index] else {
                unreachable!()
            };
            arguments.0[index] = Argument::DevicePointerArray(vec![id]);
        }
        Launch {
            symbol: entry.symbol.clone(),
            rows,
            plan,
            arguments,
            collective: false,
            persistent: false,
        }
    }

    const STREAMS: Streams = Streams {
        streams: 4,
        width: 2560,
        rank: 320,
        eps_bits: 1e-6_f32.to_bits(),
        capacity: 32,
    };

    fn hyper_backend() -> (Sm120, Vec<Entry>) {
        let mut entries = hc_entries(PROFILE, STREAMS).unwrap();
        let ngram = ngram_entry(
            PROFILE,
            NgramShape {
                orders: 3,
                heads_per_order: 8,
                head_width: 160,
                capacity: 32,
            },
            4,
            4,
        )
        .unwrap();
        // The decode form shares the query; the backend takes the
        // per-sequence form.
        entries.push(ngram_decode_entry(&ngram, 4).unwrap());
        entries.push(ngram);
        entries.push(ple_gate_entry(PROFILE, STREAMS).unwrap());
        let backend = entries
            .iter()
            .cloned()
            .fold(Sm120::new(), |backend, entry| {
                if matches!(
                    entry.query,
                    StaticQuery::Ngram(NgramQuery::HashGather { .. })
                ) {
                    backend.with_ngram(entry, EOS)
                } else {
                    backend.with_hyper_connection_row(entry)
                }
            });
        (backend, entries)
    }

    /// The end-of-document token the test n-gram entries compile in.
    const EOS: u32 = 151_643;

    fn query(entries: &[Entry], index: usize) -> StaticQuery {
        entries[index].query.clone()
    }

    #[test]
    fn hyper_connection_rows_launch_what_the_builders_plan() {
        let (backend, entries) = hyper_backend();
        let (rows, wide, width) = (32, 4 * 2560 * 2, 2560 * 2);
        let StaticQuery::HyperConnection(norm) = query(&entries, 0) else {
            unreachable!()
        };
        let prepared = backend
            .hc_grouped_norm(&norm, DType::Bf16, Need::rows(32))
            .unwrap();
        assert_eq!(prepared.info().guarantees, entries[0].invariance);
        let buffers = GroupedNormBuffers {
            x: view(1, rows * wide),
            weight: view(2, wide),
            output: view(3, rows * wide),
            live_rows: view(4, 4),
        };
        for live in [1, 7, 32] {
            assert_eq!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(live))
                    .unwrap(),
                [built_row(&entries[0], live, handles(&[1, 2, 3, 4]))]
            );
        }
        let short = GroupedNormBuffers {
            weight: view(2, wide - 2),
            ..buffers
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(33))
                .is_err()
        );

        let StaticQuery::HyperConnection(down) = query(&entries, 1) else {
            unreachable!()
        };
        let prepared = backend.hc_bottleneck(&down, Need::rows(16)).unwrap();
        let buffers = BottleneckBuffers {
            down: view(1, 16 * 320 * 2),
            output: view(2, 16 * 320 * 2),
            live_rows: view(3, 4),
        };
        assert_eq!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(16))
                .unwrap(),
            [built_row(&entries[1], 16, handles(&[1, 2, 3]))]
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(17))
                .is_err()
        );

        let StaticQuery::HyperConnection(mix) = query(&entries, 2) else {
            unreachable!()
        };
        let prepared = backend.hc_mix(&mix, Need::rows(32)).unwrap();
        let buffers = MixBuffers {
            up: view(1, rows * wide),
            normed: view(2, rows * wide),
            output: view(3, rows * width),
            live_rows: view(4, 4),
        };
        assert_eq!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(5))
                .unwrap(),
            [built_row(&entries[2], 5, handles(&[1, 2, 3, 4]))]
        );
        let short = MixBuffers {
            output: view(3, rows * width - 2),
            ..buffers
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(32))
                .is_err()
        );

        let StaticQuery::HyperConnection(inject) = query(&entries, 3) else {
            unreachable!()
        };
        let prepared = backend.hc_inject(&inject, Need::rows(32)).unwrap();
        let buffers = InjectBuffers {
            x: view(1, rows * wide),
            y: view(2, rows * width),
            inject: view(3, rows * 4 * 2),
            output: view(4, rows * wide),
            live_rows: view(5, 4),
        };
        assert_eq!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(32))
                .unwrap(),
            [built_row(&entries[3], 32, handles(&[1, 2, 3, 4, 5]))]
        );
        let short = InjectBuffers {
            inject: view(3, rows * 4 * 2 - 2),
            ..buffers
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(32))
                .is_err()
        );

        let StaticQuery::Ngram(gate) = query(&entries, 6) else {
            unreachable!()
        };
        let prepared = backend.ple_gate(&gate, Need::rows(32)).unwrap();
        let buffers = PleGateBuffers {
            key: view(1, rows * wide),
            query: view(2, rows * wide),
            value: view(3, rows * width),
            output: view(4, rows * wide),
            live_rows: view(5, 4),
        };
        assert_eq!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(9))
                .unwrap(),
            [built_row(&entries[6], 9, handles(&[1, 2, 3, 4, 5]))]
        );
        let short = PleGateBuffers {
            value: view(3, 9 * width - 2),
            ..buffers
        };
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(9))
                .is_err()
        );
    }

    #[test]
    fn grouped_norms_take_the_entry_of_their_weight_dtype() {
        let f32_norm = GroupedRmsNorm {
            reduction: NormReduction::ShuffleDown,
            weight_dtype: DType::F32,
            group_width: 2560,
            groups: 4,
        };
        let symbol = "chop_row_test_hc_norm_f32_v1";
        let (backend, entries) = hyper_backend();
        let backend = backend.with_grouped_norm(f32_norm, symbol);
        let StaticQuery::HyperConnection(norm) = query(&entries, 0) else {
            unreachable!()
        };
        let (rows, wide) = (32, 4 * 2560);
        let buffers = GroupedNormBuffers {
            x: view(1, rows * wide * 2),
            weight: view(2, wide * 4),
            output: view(3, rows * wide * 2),
            live_rows: view(4, 4),
        };
        // F32 weights take the grouped norm entry, as a model plans it at
        // the step's rows.
        let prepared = backend
            .hc_grouped_norm(&norm, DType::F32, Need::rows(32))
            .unwrap();
        assert_eq!(prepared.info().symbols, [(symbol.to_owned(), 0)]);
        assert_eq!(prepared.info().guarantees, Invariance::NONE);
        for live in [1, 9, 32] {
            let arguments = ArgumentList(handles(&[1, 2, 3, 4]));
            assert_eq!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(live))
                    .unwrap(),
                [Launch {
                    symbol: symbol.into(),
                    rows: live,
                    plan: f32_norm.plan(live, live).unwrap(),
                    arguments,
                    collective: false,
                    persistent: false,
                }]
            );
        }
        // The F32 weights are twice a BF16 weight's bytes.
        let bf16_sized = GroupedNormBuffers {
            weight: view(2, wide * 2),
            ..buffers
        };
        assert!(
            prepared
                .launches(&bf16_sized, View::EMPTY, Counts::rows(4))
                .is_err()
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(33))
                .is_err()
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(0))
                .is_err()
        );
        // BF16 weights keep the hyper-connection row and its size check.
        let prepared = backend
            .hc_grouped_norm(&norm, DType::Bf16, Need::rows(32))
            .unwrap();
        assert_eq!(prepared.info().symbols, [(entries[0].symbol.clone(), 0)]);
        assert!(
            prepared
                .launches(&bf16_sized, View::EMPTY, Counts::rows(4))
                .is_ok()
        );
        // A BF16 entry does not serve F32 weights, nor F32 entries BF16 ones.
        let (bf16_only, _) = hyper_backend();
        assert!(matches!(
            bf16_only.hc_grouped_norm(&norm, DType::F32, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
        let f32_only = Sm120::new().with_grouped_norm(f32_norm, symbol);
        assert!(matches!(
            f32_only.hc_grouped_norm(&norm, DType::Bf16, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
        // Other dtypes, capacities past the query's, and another epsilon.
        assert!(matches!(
            backend.hc_grouped_norm(&norm, DType::Fp8E4M3, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            backend.hc_grouped_norm(&norm, DType::F32, Need::rows(33)),
            Err(Refused::Rule(_))
        ));
        let H::GroupedNorm {
            streams,
            width,
            capacity,
            ..
        } = norm
        else {
            unreachable!()
        };
        let other_eps = H::GroupedNorm {
            streams,
            width,
            eps_bits: 1e-5_f32.to_bits(),
            capacity,
        };
        assert!(matches!(
            backend.hc_grouped_norm(&other_eps, DType::F32, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
    }

    #[test]
    fn oversized_capacities_are_refused_before_planning() {
        let rule = |result: Result<(), Refused>| matches!(result, Err(Refused::Rule(_)));
        let shard = VOCAB / 4;
        let (partial, last) = argmax_entries(shard, 4);
        let (backend, entries) = hyper_backend();
        let backend = backend
            .with_row(embedding_entry(
                PROFILE,
                "embedding_full",
                VOCAB,
                WIDTH,
                false,
            ))
            .with_row(logits_entry(PROFILE, "argmax_full", VOCAB, 1))
            .with_row(partial)
            .with_row(last);
        let embedding = RowQuery::EmbeddingGather {
            vocab: VOCAB,
            width: WIDTH,
        };
        let argmax = RowQuery::LogitsArgmax { vocab: VOCAB };
        let partial = partial_query(shard, 4);
        let last = final_query(4);
        let hyper = |index| {
            let StaticQuery::HyperConnection(query) = query(&entries, index) else {
                unreachable!()
            };
            query
        };
        let StaticQuery::Ngram(ngram) = query(&entries, 5) else {
            unreachable!()
        };
        let StaticQuery::Ngram(gate) = query(&entries, 6) else {
            unreachable!()
        };
        // An F32 grouped norm whose query allows any row count: SM120's row
        // bound refuses it before any plan is built.
        let f32_norm = GroupedRmsNorm {
            reduction: NormReduction::ShuffleDown,
            weight_dtype: DType::F32,
            group_width: 2560,
            groups: 4,
        };
        let backend = backend.with_grouped_norm(f32_norm, "chop_row_test_hc_norm_f32_v1");
        let unbounded = H::GroupedNorm {
            streams: 4,
            width: 2560,
            eps_bits: 1e-6_f32.to_bits(),
            capacity: u32::MAX,
        };
        for capacity in [MAX_ROWS + 1, MAX_CAPACITY + 1, u32::MAX] {
            let need = Need::rows(capacity);
            assert!(rule(
                backend.embedding_gather(&embedding, 0, need).map(drop)
            ));
            assert!(rule(backend.logits_top_k(&argmax, need).map(drop)));
            assert!(rule(
                backend.sharded_argmax_partial(&partial, 1, need).map(drop)
            ));
            assert!(rule(backend.sharded_argmax_final(&last, need).map(drop)));
            assert!(rule(
                backend
                    .hc_grouped_norm(&hyper(0), DType::Bf16, need)
                    .map(drop)
            ));
            assert!(rule(backend.hc_bottleneck(&hyper(1), need).map(drop)));
            assert!(rule(backend.hc_mix(&hyper(2), need).map(drop)));
            assert!(rule(backend.hc_inject(&hyper(3), need).map(drop)));
            assert!(rule(
                backend.ngram_hash_gather(&ngram, EOS, 0, need).map(drop)
            ));
            assert!(rule(backend.ple_gate(&gate, need).map(drop)));
            assert!(rule(
                backend
                    .hc_grouped_norm(&unbounded, DType::F32, need)
                    .map(drop)
            ));
        }
        // The bound itself is served.
        assert!(
            backend
                .hc_grouped_norm(&unbounded, DType::F32, Need::rows(MAX_ROWS))
                .is_ok()
        );
    }

    #[test]
    fn ngram_launches_the_per_sequence_entry() {
        let (backend, entries) = hyper_backend();
        let StaticQuery::Ngram(query) = query(&entries, 5) else {
            unreachable!()
        };
        let prepared = backend
            .ngram_hash_gather(&query, EOS, 3, Need::rows(32))
            .unwrap();
        assert_eq!(prepared.info().symbols, [(entries[5].symbol.clone(), 0)]);
        assert_eq!(prepared.info().guarantees, Invariance::PREFILL);
        let (rows, local) = (32, 4 * 160);
        let buffers = NgramBuffers {
            tokens: view(1, rows * 4),
            history: view(2, 8),
            multipliers: view(3, 3 * 8),
            sizes: view(4, 16 * 8),
            offsets: view(5, 16 * 8),
            table: view(6, 1000 * 160),
            scale: view(7, 2),
            output: view(8, rows * local * 2),
            row_ids: view(9, rows * 4 * 8),
            history_out: view(10, 8),
            live_rows: view(11, 4),
            table_start: 3000,
            table_rows: 1000,
        };
        let mut arguments = handles(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        arguments.extend([Argument::U32(3), Argument::U64(3000), Argument::U64(1000)]);
        assert_eq!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(12))
                .unwrap(),
            [built_row(&entries[5], 12, arguments)]
        );
        for short in [
            NgramBuffers {
                table: view(6, 1000 * 160 - 1),
                ..buffers
            },
            NgramBuffers {
                sizes: view(4, 15 * 8),
                ..buffers
            },
            NgramBuffers {
                row_ids: view(9, 12 * 4 * 8 - 8),
                ..buffers
            },
            NgramBuffers {
                history_out: view(2, 8),
                ..buffers
            },
        ] {
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(12))
                    .is_err()
            );
        }
        // Four ranks hold the sixteen heads.
        assert!(matches!(
            backend.ngram_hash_gather(&query, EOS, 4, Need::rows(32)),
            Err(Refused::Rule(_))
        ));
        // Another end-of-document token than the entry compiles in.
        assert!(matches!(
            backend.ngram_hash_gather(&query, EOS + 1, 0, Need::rows(32)),
            Err(Refused::Rule(rule)) if rule.contains("end-of-document")
        ));
        // Orders past 8 are math the contract takes and SM120 refuses.
        let orders = NgramQuery::HashGather {
            orders: 9,
            heads_per_order: 2,
            head_width: 160,
            local_heads: 4,
            capacity: 32,
        };
        assert!(orders.validate().is_ok());
        assert!(matches!(
            backend.ngram_hash_gather(&orders, EOS, 0, Need::rows(32)),
            Err(Refused::Rule(rule)) if rule.contains("no SM120")
        ));
        // An n-gram entry registered without its token serves no gather.
        let unregistered = Sm120::new().with_hyper_connection_row(entries[5].clone());
        assert!(matches!(
            unregistered.ngram_hash_gather(&query, EOS, 0, Need::rows(32)),
            Err(Refused::Rule(rule)) if rule.contains("no SM120")
        ));
    }

    #[test]
    fn hyper_connection_rows_refuse_what_they_cannot_serve() {
        let (backend, entries) = hyper_backend();
        let StaticQuery::HyperConnection(norm) = query(&entries, 0) else {
            unreachable!()
        };
        // Past the compiled capacity: an SM120 entry limit.
        assert!(matches!(
            backend.hc_grouped_norm(&norm, DType::Bf16, Need::rows(33)),
            Err(Refused::Rule(_))
        ));
        // Exactness past the declaration.
        assert!(matches!(
            backend.hc_grouped_norm(
                &norm,
                DType::Bf16,
                Need::rows(32).with_exact(Invariance::any_rows(64))
            ),
            Err(Refused::Inexact { .. })
        ));
        // Another kind, the collective form and an unserved geometry.
        assert!(matches!(
            backend.hc_mix(&norm, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
        let collective = H::AllReduceInjectNorm {
            ranks: 4,
            streams: 4,
            width: 2560,
            eps_bits: 1e-6_f32.to_bits(),
            capacity: 32,
        };
        assert!(matches!(
            backend.hc_inject(&collective, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
        let other = H::GroupedNorm {
            streams: 2,
            width: 2560,
            eps_bits: 1e-6_f32.to_bits(),
            capacity: 32,
        };
        assert!(matches!(
            backend.hc_grouped_norm(&other, DType::Bf16, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
        // The n-gram's decode form alone serves no chunk.
        let decode = entries
            .iter()
            .find(|e| e.symbol.ends_with("ngram_decode_v1"));
        let decode = decode.cloned().unwrap();
        let StaticQuery::Ngram(query) = decode.query.clone() else {
            unreachable!()
        };
        assert!(matches!(
            Sm120::new()
                .with_ngram(decode, EOS)
                .ngram_hash_gather(&query, EOS, 0, Need::rows(4)),
            Err(Refused::Rule(_))
        ));
    }

    /// A row registration of a permutation case.
    #[derive(Clone)]
    enum Registered {
        Row(RowInstantiation),
        Shard(RowInstantiation),
        Hyper(Entry),
        Ngram(Entry, u32),
        Norm(GroupedRmsNorm, &'static str),
    }

    fn register(order: &[Registered]) -> Sm120 {
        order
            .iter()
            .cloned()
            .fold(Sm120::new(), |backend, registered| match registered {
                Registered::Row(entry) => backend.with_row(entry),
                Registered::Shard(entry) => backend.with_sharded_embedding(entry),
                Registered::Hyper(entry) => backend.with_hyper_connection_row(entry),
                Registered::Ngram(entry, eos) => backend.with_ngram(entry, eos),
                Registered::Norm(norm, symbol) => backend.with_grouped_norm(norm, symbol),
            })
    }

    /// The symbols `prepare` prepares over every order of `registered`, or
    /// the refusal, the same in every order.
    fn in_every_order(
        registered: &[Registered],
        prepare: impl Fn(&Sm120) -> Result<OpInfo, Refused>,
    ) -> Result<Vec<String>, Refused> {
        let results = orders(registered)
            .iter()
            .map(|order| {
                prepare(&register(order))
                    .map(|info| info.symbols.into_iter().map(|(symbol, _)| symbol).collect())
            })
            .collect::<Vec<_>>();
        assert!(results.iter().all(|result| *result == results[0]));
        results[0].clone()
    }

    fn twin<T: Clone>(entry: &T, rename: impl Fn(&mut T)) -> T {
        let mut twin = entry.clone();
        rename(&mut twin);
        twin
    }

    /// Whether `result` refuses a tie of `symbols`, named in sorted order.
    fn tied(result: Result<Vec<String>, Refused>, mut symbols: [&str; 2]) -> bool {
        symbols.sort_unstable();
        matches!(result, Err(Refused::Rule(rule)) if rule.contains(&symbols.join(" and ")))
    }

    #[test]
    fn row_entries_prepare_alike_in_every_order() {
        let query = RowQuery::EmbeddingGather {
            vocab: VOCAB,
            width: WIDTH,
        };
        let full = embedding_entry(PROFILE, "embedding_full", VOCAB, WIDTH, false);
        let other = twin(&full, |e| e.symbol = "chop_row_embedding_twin_v1".into());
        let shard = embedding_entry(PROFILE, "embedding_shard", VOCAB, WIDTH, true);
        let embedding = |offset| {
            move |backend: &Sm120| {
                backend
                    .embedding_gather(&query, offset, Need::rows(4))
                    .map(|op| op.info().clone())
            }
        };
        // Offset zero takes the whole-table entry over the sharded one, and
        // a duplicate registration is one entry.
        let registered = [
            Registered::Row(full.clone()),
            Registered::Shard(shard.clone()),
            Registered::Row(full.clone()),
        ];
        assert_eq!(
            in_every_order(&registered, embedding(0)),
            Ok(vec![full.symbol.clone()])
        );
        assert_eq!(
            in_every_order(&registered, embedding(4096)),
            Ok(vec![shard.symbol.clone()])
        );
        // Two whole-table entries of one query are a tie; a sharded one at
        // another offset is not.
        let registered = [
            Registered::Row(full.clone()),
            Registered::Row(other.clone()),
            Registered::Shard(shard.clone()),
        ];
        assert!(tied(
            in_every_order(&registered, embedding(0)),
            [&full.symbol, &other.symbol]
        ));
        assert_eq!(
            in_every_order(&registered, embedding(4096)),
            Ok(vec![shard.symbol.clone()])
        );
    }

    #[test]
    fn hyper_connection_rows_prepare_alike_in_every_order() {
        let (_, entries) = hyper_backend();
        let need = Need::rows(4);
        // Grouped norms of two reductions are a tie; the op does not choose
        // the reduction order.
        let StaticQuery::HyperConnection(norm) = query(&entries, 0) else {
            unreachable!()
        };
        let down = GroupedRmsNorm {
            reduction: NormReduction::ShuffleDown,
            weight_dtype: DType::F32,
            group_width: 2560,
            groups: 4,
        };
        let xor = GroupedRmsNorm {
            reduction: NormReduction::XorThenSerial,
            ..down
        };
        let grouped = |backend: &Sm120| {
            backend
                .hc_grouped_norm(&norm, DType::F32, need)
                .map(|op| op.info().clone())
        };
        let (a, b) = ("chop_row_test_norm_a_v1", "chop_row_test_norm_b_v1");
        let duplicate = [Registered::Norm(down, a), Registered::Norm(down, a)];
        assert_eq!(in_every_order(&duplicate, grouped), Ok(vec![a.to_owned()]));
        for registered in [
            [Registered::Norm(down, a), Registered::Norm(xor, b)],
            [Registered::Norm(down, b), Registered::Norm(xor, a)],
        ] {
            assert!(tied(in_every_order(&registered, grouped), [a, b]));
        }
        // A reduction alone tells same-symbol entries apart, so they tie.
        let registered = [Registered::Norm(down, a), Registered::Norm(xor, a)];
        assert!(tied(in_every_order(&registered, grouped), [a, a]));
        // Hyper-connection rows of one query and two symbols tie.
        let StaticQuery::HyperConnection(bottleneck) = query(&entries, 1) else {
            unreachable!()
        };
        let other = twin(&entries[1], |e| e.symbol = "chop_row_test_twin_v1".into());
        let prepare = |backend: &Sm120| {
            backend
                .hc_bottleneck(&bottleneck, need)
                .map(|op| op.info().clone())
        };
        let registered = [
            Registered::Hyper(entries[1].clone()),
            Registered::Hyper(entries[1].clone()),
        ];
        assert_eq!(
            in_every_order(&registered, prepare),
            Ok(vec![entries[1].symbol.clone()])
        );
        let registered = [
            Registered::Hyper(entries[1].clone()),
            Registered::Hyper(other.clone()),
        ];
        assert!(tied(
            in_every_order(&registered, prepare),
            [&other.symbol, &entries[1].symbol]
        ));
        // n-gram entries are told apart by their compiled token, and tie
        // at one token.
        let StaticQuery::Ngram(ngram) = query(&entries, 5) else {
            unreachable!()
        };
        let other = twin(&entries[5], |e| {
            e.symbol = "chop_row_test_ngram_twin_v1".into()
        });
        let gather = |backend: &Sm120| {
            backend
                .ngram_hash_gather(&ngram, EOS, 0, need)
                .map(|op| op.info().clone())
        };
        let registered = [
            Registered::Ngram(entries[5].clone(), EOS),
            Registered::Ngram(other.clone(), EOS + 1),
        ];
        assert_eq!(
            in_every_order(&registered, gather),
            Ok(vec![entries[5].symbol.clone()])
        );
        let registered = [
            Registered::Ngram(entries[5].clone(), EOS),
            Registered::Ngram(other.clone(), EOS),
        ];
        assert!(tied(
            in_every_order(&registered, gather),
            [&entries[5].symbol, &other.symbol]
        ));
    }

    /// R1 finding 4: an entry registered again under another profile or
    /// with another claim launches the same, so it is one entry, and the
    /// one that stands for both does not depend on the order.
    #[test]
    fn launch_identical_registrations_are_one_entry() {
        let query = RowQuery::EmbeddingGather {
            vocab: VOCAB,
            width: WIDTH,
        };
        let full = embedding_entry(PROFILE, "embedding_full", VOCAB, WIDTH, false);
        let profiled = twin(&full, |e| {
            e.profile = Profile::synthetic(crate::ProfileNames {
                name: "row-backend-other",
                tag: "row_backend_other",
            })
        });
        let claimed = twin(&full, |e| e.invariance = crate::row::row_local());
        assert_ne!(full.invariance, claimed.invariance);
        let registered = [
            Registered::Row(full.clone()),
            Registered::Row(profiled),
            Registered::Row(claimed.clone()),
        ];
        let infos = orders(&registered)
            .iter()
            .map(|order| {
                register(order)
                    .embedding_gather(&query, 0, Need::rows(4))
                    .map(|op| op.info().clone())
            })
            .collect::<Vec<_>>();
        assert!(infos.iter().all(|info| *info == infos[0]));
        assert_eq!(infos[0].as_ref().unwrap().symbols[0].0, full.symbol);
        // A need only the claiming registration keeps takes it.
        let exact = Need::rows(4).with_exact(claimed.invariance);
        for order in orders(&registered) {
            let info = register(&order)
                .embedding_gather(&query, 0, exact)
                .unwrap()
                .info()
                .clone();
            assert_eq!(info.guarantees, claimed.invariance);
        }
    }

    /// Registrations of one launch claiming granules 64 and 128 both keep a
    /// need at 128; together they claim 128, their meet, in either order.
    #[test]
    fn granule_claims_of_one_launch_meet_at_their_common_multiple() {
        let query = RowQuery::EmbeddingGather {
            vocab: VOCAB,
            width: WIDTH,
        };
        let granule = |n| Invariance::NONE.with_chunk(Chunk::Granule(n));
        let full = embedding_entry(PROFILE, "embedding_full", VOCAB, WIDTH, false);
        let at64 = twin(&full, |e| e.invariance = granule(64));
        let at128 = twin(&full, |e| e.invariance = granule(128));
        let need = Need::rows(1).with_exact(granule(128));
        let registered = [Registered::Row(at64), Registered::Row(at128)];
        for order in orders(&registered) {
            let info = register(&order)
                .embedding_gather(&query, 0, need)
                .unwrap()
                .info()
                .clone();
            assert_eq!(info.guarantees, granule(128));
        }
    }

    /// The engine row, grouped norm and hyper-connection keys hold what
    /// their entries launch, every field changed alone: the code (ABI and
    /// body), grid, threads and symbol, a norm's reduction, an entry's
    /// grouping; never the profile, the claims, or the width the compile
    /// key records (the auditor's embedding twin), and the query selects.
    #[test]
    fn launch_keys_hold_what_rows_launch() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let other = Profile::synthetic(crate::ProfileNames {
            name: "row-backend-other",
            tag: "row_backend_other",
        });
        // An engine row: the embedding gather.
        let gather = RowQuery::EmbeddingGather {
            vocab: VOCAB,
            width: WIDTH,
        };
        let full = embedding_entry(PROFILE, "embedding_full", VOCAB, WIDTH, false);
        let row = u64::from(WIDTH) * 2;
        let buffers = EmbeddingBuffers {
            table: view(1, u64::from(VOCAB) * row),
            ids: view(2, 32 * 4),
            output: view(3, 32 * row),
            live_rows: view(4, 4),
            table_rows: VOCAB,
        };
        let names = field_names!(RowInstantiation {
            profile,
            symbol,
            query,
            width,
            grid,
            threads,
            params,
            body,
            invariance
        });
        let fields: Vec<Field<RowInstantiation>> = vec![
            (
                "profile",
                Class::Inert,
                Box::new(move |e| e.profile = other),
            ),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_row_other_v1".into()),
            ),
            (
                "query",
                Class::Selects,
                Box::new(|e| {
                    e.query = RowQuery::EmbeddingGather {
                        vocab: VOCAB,
                        width: WIDTH * 2,
                    }
                }),
            ),
            ("width", Class::Inert, Box::new(|e| e.width *= 2)),
            (
                "grid",
                Class::Launch,
                Box::new(|e| e.grid = Grid::RowsHeads(2)),
            ),
            ("threads", Class::Launch, Box::new(|e| e.threads *= 2)),
            ("params", Class::Launch, Box::new(|e| e.params.swap(0, 1))),
            ("body", Class::Launch, Box::new(|e| e.body.push_str("//"))),
            (
                "invariance",
                Class::Inert,
                Box::new(|e| e.invariance = crate::row::row_local()),
            ),
        ];
        keys_follow_launches(&full, names, fields, row_launch, |entry| {
            let op = Sm120::new().with_row(entry.clone()).embedding_gather(
                &gather,
                0,
                Need::rows(32),
            )?;
            let code = format!("{:?} {} {}", entry.params, entry.body, entry.threads);
            let launches = [1, 32].map(|rows| {
                op.launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap()
            });
            Ok((code, launches))
        });
        // A grouped norm: its symbol and reduction; the op selects the rest.
        let norm = GroupedRmsNorm {
            reduction: NormReduction::ShuffleDown,
            weight_dtype: DType::F32,
            group_width: 2560,
            groups: 4,
        };
        let (_, entries) = hyper_backend();
        let StaticQuery::HyperConnection(op) = query(&entries, 0) else {
            unreachable!()
        };
        let (rows, wide) = (32, 4 * 2560);
        let buffers = GroupedNormBuffers {
            x: view(1, rows * wide * 2),
            weight: view(2, wide * 4),
            output: view(3, rows * wide * 2),
            live_rows: view(4, 4),
        };
        let mut names = field_names!(GroupedRmsNorm {
            reduction,
            weight_dtype,
            group_width,
            groups
        });
        names.push("symbol");
        let fields: Vec<Field<(GroupedRmsNorm, String)>> = vec![
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.1 = "chop_row_other_norm_v1".into()),
            ),
            (
                "reduction",
                Class::Launch,
                Box::new(|e| e.0.reduction = NormReduction::XorThenSerial),
            ),
            (
                "weight_dtype",
                Class::Selects,
                Box::new(|e| e.0.weight_dtype = DType::Bf16),
            ),
            (
                "group_width",
                Class::Selects,
                Box::new(|e| e.0.group_width *= 2),
            ),
            ("groups", Class::Selects, Box::new(|e| e.0.groups = 8)),
        ];
        let base = (norm, "chop_row_test_hc_norm_f32_v1".to_owned());
        keys_follow_launches(
            &base,
            names,
            fields,
            |(norm, symbol)| norm_launch(norm, symbol),
            |(norm, symbol)| {
                let prepared = Sm120::new()
                    .with_grouped_norm(*norm, symbol.clone())
                    .hc_grouped_norm(&op, DType::F32, Need::rows(32))?;
                let launches = [1, 32].map(|rows| {
                    prepared
                        .launches(&buffers, View::EMPTY, Counts::rows(rows))
                        .unwrap()
                });
                Ok((format!("{:?}", norm.entry(symbol)), launches))
            },
        );
        // A hyper-connection row.
        let entry = entries[0].clone();
        let buffers = GroupedNormBuffers {
            x: view(1, rows * wide * 2),
            weight: view(2, wide * 2),
            output: view(3, rows * wide * 2),
            live_rows: view(4, 4),
        };
        let names = field_names!(Entry {
            profile,
            query,
            symbol,
            params,
            body,
            groups,
            items,
            invariance
        });
        let down = query(&entries, 1);
        let fields: Vec<Field<Entry>> = vec![
            (
                "profile",
                Class::Inert,
                Box::new(move |e| e.profile = other),
            ),
            (
                "query",
                Class::Selects,
                Box::new(move |e| e.query = down.clone()),
            ),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_row_other_hc_v1".into()),
            ),
            ("params", Class::Launch, Box::new(|e| e.params.swap(0, 1))),
            ("body", Class::Launch, Box::new(|e| e.body.push_str("//"))),
            ("groups", Class::Selects, Box::new(|e| e.groups *= 2)),
            ("items", Class::Selects, Box::new(|e| e.items = !e.items)),
            (
                "invariance",
                Class::Inert,
                Box::new(|e| e.invariance = Invariance::NONE),
            ),
        ];
        keys_follow_launches(&entry, names, fields, hyper_launch, |e| {
            let prepared = Sm120::new()
                .with_hyper_connection_row(e.clone())
                .hc_grouped_norm(&op, DType::Bf16, Need::rows(32))?;
            let launches = [1, 32].map(|rows| {
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap()
            });
            Ok((format!("{:?} {}", e.params, e.body), launches))
        });
    }

    /// The auditor's case: embedding registrations differing only in the
    /// width their compile key records are one entry in every order.
    #[test]
    fn embeddings_differing_in_compile_key_width_are_one() {
        let query = RowQuery::EmbeddingGather {
            vocab: VOCAB,
            width: WIDTH,
        };
        let full = embedding_entry(PROFILE, "embedding_full", VOCAB, WIDTH, false);
        let wide = twin(&full, |e| e.width *= 2);
        for order in orders(&[Registered::Row(full.clone()), Registered::Row(wide)]) {
            let info = register(&order)
                .embedding_gather(&query, 0, Need::rows(4))
                .unwrap()
                .info()
                .clone();
            assert_eq!(info.symbols[0].0, full.symbol);
        }
    }
}
