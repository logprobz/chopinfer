//! Sinkhorn-mixed hyper-connections through the model's entries: the
//! mix-collapse, expand and stream head of `sinkhorn_hc`, and the comm
//! epilogue of `comm::sinkhorn_split` after the F32 all-reduce. Each
//! prepared op plans every row count up to its capacity as the builders
//! plan it; the kernels read the live row count on the device.
//!
//! An op runs the entry [`fitting`] chooses, whatever the order the model
//! registered its entries in. The mix-collapse runs the model's split entry
//! (`sinkhorn_hc::split`) when one serves the op, else its one-CTA entry;
//! at the row counts SM120's tile schedule gives a tile
//! (`sinkhorn_hc::tiled::served_prefill_mix`, kernel-lab timing on the
//! served GPU) it runs the model's tiled entry of that tile
//! (`sinkhorn_hc::tiled`) when one serves the op. All three add the
//! projection in 128-K chains from zero in K order and write equal bits, so
//! they share one implementation; the CTA split and the tile are SM120's
//! choice. The mix-collapse with its input norm (`MixCollapseNorm`) runs
//! the model's split entry fused with the tree norm
//! (`sinkhorn_hc::mix_norm_tree`): the split's outputs and the row
//! engine's tree norm of the collapsed row, bit for bit. The comm
//! epilogue's outputs equal the unfused round, expand, mix-collapse and
//! HF-ordered input norm bit for bit.
//!
//! Each op claims what its entries declare (`sinkhorn_hc`'s module docs:
//! one-row bits up to the capacity, any chunk split, row independence).
//! The mix-collapse launches different entries at different row counts;
//! each declares the one-CTA entry's one-row bits at every row count it
//! runs, so the op keeps one-row bits at every row count
//! (`Invariance::routed` over the one-CTA reference) and, handing no state
//! to a later launch, any chunk split. The comm epilogue claims what its
//! launches declare together: its all-reduce's claims (the widen row's too
//! after a BF16-partials all-reduce) and its split entry's
//! (`sinkhorn_split::Entry::invariance`). Every op is refused past
//! the SM120 kernels' limits (`sinkhorn_hc::limits`) before an entry is
//! looked up.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::collective::AllReduceBuffers;
use kernel_core::contract::{Argument, CommQuery, DType, LaunchPlan, Refusal};
use kernel_core::properties::Invariance;
use kernel_core::sinkhorn_hyper_connection::{
    MixCollapseBuffers, MixCollapseNormBuffers, MixScratch, MixerWeights,
    ReduceExpandMixNormBuffers, SinkhornHyperConnectionQuery as Query, SinkhornMix,
    StreamExpandBuffers, StreamHeadBuffers,
};

use super::check_plan;
use super::select;
use crate::comm::sinkhorn_epilogue::EpilogueBuffers;
use crate::comm::sinkhorn_split::{self, SplitBuffers};
use crate::sinkhorn_hc::{self, Op, mix_norm_tree, split, tiled};

/// The Sinkhorn entries of a model's selection.
#[derive(Default)]
pub(super) struct Selection {
    /// One-CTA mix-collapse, expand and head entries.
    pub(super) entries: Vec<sinkhorn_hc::Entry>,
    /// Split mix-collapse entries, which serve in place of the one-CTA
    /// entry.
    pub(super) split: Vec<split::Entry>,
    /// Tiled mix-collapse entries, which serve their tile's row counts in
    /// place of the split or one-CTA entry.
    pub(super) tiled: Vec<tiled::Entry>,
    /// Split mix-collapse entries fused with the input norm's tree.
    pub(super) mix_norm: Vec<mix_norm_tree::Entry>,
    /// Comm epilogues over the all-reduce's F32 sums.
    pub(super) epilogues: Vec<sinkhorn_split::Entry>,
    /// The lagged engine's mix-collapse, F32 expand and weighted head
    /// entries (`super::sinkhorn_lagged`).
    pub(super) lagged: Vec<sinkhorn_hc::lagged::Entry>,
    pub(super) lagged_split: Vec<sinkhorn_hc::lagged_split::Entry>,
    /// Split lagged mixers that serve only an op they hold whole: one
    /// prepared for at most their capacity.
    pub(super) lagged_split_bounded: Vec<sinkhorn_hc::lagged_split::Entry>,
    pub(super) lagged_tiled: Vec<sinkhorn_hc::lagged_tiled::Entry>,
    pub(super) lagged_warp: Vec<sinkhorn_hc::lagged_warp::Entry>,
    pub(super) lagged_prefill: Vec<(sinkhorn_hc::prefill_warp::Entry, Vec<u32>)>,
}

/// The mix-collapse's numerics, whichever entry runs it.
const MIX_COLLAPSE: &str = "sm120.sinkhorn.mix-collapse.v1[chain=128]";
/// The mix-collapse's numerics, then the input norm's 256-lane tree.
const MIX_COLLAPSE_NORM: &str = "sm120.sinkhorn.mix-collapse-norm-tree.v1[chain=128]";
/// The comm epilogue's numerics after its all-reduce's.
const EXPAND_MIX_NORM: &str = "sm120.sinkhorn.expand-mix-norm.v1[chain=128]";

/// The row bound of `query`.
fn capacity(query: Query) -> u32 {
    match query {
        Query::MixCollapse { capacity, .. }
        | Query::MixCollapseNorm { capacity, .. }
        | Query::Expand { capacity, .. }
        | Query::ExpandF32 { capacity, .. }
        | Query::Head { capacity, .. }
        | Query::AllReduceExpandMixNorm { capacity, .. } => capacity,
    }
}

/// `query` with no row bound: an entry serves the queries equal to its own
/// but for a capacity within its own.
pub(super) fn uncapped(query: Query) -> Query {
    match query {
        Query::MixCollapse { mix, .. } => Query::MixCollapse { mix, capacity: 0 },
        Query::MixCollapseNorm {
            mix,
            rmsnorm_eps_bits,
            ..
        } => Query::MixCollapseNorm {
            mix,
            rmsnorm_eps_bits,
            capacity: 0,
        },
        Query::Expand { streams, width, .. } => Query::Expand {
            streams,
            width,
            capacity: 0,
        },
        Query::ExpandF32 { streams, width, .. } => Query::ExpandF32 {
            streams,
            width,
            capacity: 0,
        },
        Query::Head {
            streams,
            width,
            mode,
            ..
        } => Query::Head {
            streams,
            width,
            mode,
            capacity: 0,
        },
        Query::AllReduceExpandMixNorm {
            ranks,
            dtype,
            mix,
            rmsnorm_eps_bits,
            norm,
            ..
        } => Query::AllReduceExpandMixNorm {
            ranks,
            dtype,
            mix,
            rmsnorm_eps_bits,
            norm,
            capacity: 0,
        },
    }
}

/// Why no entry serves `op`: the refusal [`fitting`] found, or that no
/// entry runs its geometry.
fn unserved(refusal: Option<Refused>, op: &str) -> Refused {
    refusal
        .unwrap_or_else(|| Refused::Rule(format!("no SM120 Sinkhorn {op} entry serves the query")))
}

/// Refuses launches past `op`'s row bound, before any entry is chosen.
pub(super) fn check_need(op: Query, need: Need) -> Result<(), Refused> {
    let bound = capacity(op);
    if need.capacity == 0 || need.capacity > bound {
        return Err(Refused::Rule(format!(
            "a Sinkhorn op serves 1 to its query's {bound} rows"
        )));
    }
    Ok(())
}

/// An entry of the selection that runs an op's geometry: the capacity its
/// builder accepts it at, or the builder's refusal; its symbol; and what it
/// launches (its symbol and the instantiation and argument shape it
/// launches with, without its profile).
pub(super) struct Candidate<E> {
    entry: E,
    capacity: Result<u32, Refusal>,
    symbol: String,
    launches: String,
    /// What the entry declares (`Candidate::claims`), or nothing.
    guarantees: Invariance,
}

impl<E> Candidate<E> {
    pub(super) fn new(
        entry: E,
        symbol: &str,
        launches: String,
        capacity: Result<u32, Refusal>,
    ) -> Self {
        Candidate {
            entry,
            capacity,
            symbol: symbol.to_owned(),
            launches,
            guarantees: Invariance::NONE,
        }
    }

    /// The candidate declaring `guarantees`.
    pub(super) fn claims(self, guarantees: Invariance) -> Self {
        Candidate { guarantees, ..self }
    }
}

/// What [`fitting`] found among one kind of entry.
pub(super) enum Fitting<E> {
    Chosen(E),
    /// No entry serves the op: none runs its geometry (`None`), or each
    /// that does is refused by its builder or holds too few rows.
    Unserved(Option<Refused>),
}

/// The entry of `candidates` that serves `need`, whatever the order the
/// model registered them in: of the entries their builders accept, the
/// one with the least capacity that holds the prepared rows
/// (`select::choose`). The query's capacity only bounds the need: a plan
/// depends on its bucket's rows alone. Entries of equal launches are one
/// entry; different entries tied at that capacity would launch
/// differently, so they are refused by name in sorted order. A need with
/// claims the entries' declarations do not keep is refused as inexact
/// before any tie.
pub(super) fn fitting<E: Clone + std::fmt::Debug>(
    mut candidates: Vec<Candidate<E>>,
    need: Need,
    op: &str,
) -> Result<Fitting<E>, Refused> {
    let rows = need.capacity;
    candidates.sort_by(|a, b| (&a.symbol, &a.launches).cmp(&(&b.symbol, &b.launches)));
    let Some(first) = candidates.first() else {
        return Ok(Fitting::Unserved(None));
    };
    let eligible: Vec<_> = candidates
        .iter()
        .filter_map(|c| c.capacity.as_ref().ok().map(|&capacity| (c, capacity)))
        .collect();
    let Some(widest) = eligible.iter().map(|&(_, capacity)| capacity).max() else {
        let refusal = first.capacity.expect_err("no entry is eligible");
        return Ok(Fitting::Unserved(Some(refusal.into())));
    };
    if !eligible.iter().any(|&(_, capacity)| capacity >= rows) {
        return Ok(Fitting::Unserved(Some(Refused::Rule(format!(
            "{rows} rows exceed every SM120 Sinkhorn {op} entry's capacity, at most {widest}"
        )))));
    }
    let chosen = select::choose(
        &format!("Sinkhorn {op}"),
        eligible.into_iter().map(|(candidate, capacity)| {
            select::Candidate::new(
                &candidate.entry,
                candidate.symbol.clone(),
                candidate.launches.clone(),
                candidate.guarantees,
            )
            .capacity(capacity)
        }),
        need,
    )?;
    Ok(Fitting::Chosen(chosen.entry.clone()))
}

/// The chosen entry, or why none serves the op.
pub(super) fn chosen<E>(fitting: Fitting<E>, op: &str) -> Result<E, Refused> {
    match fitting {
        Fitting::Chosen(entry) => Ok(entry),
        Fitting::Unserved(refusal) => Err(unserved(refusal, op)),
    }
}

/// The plan of each row count from one to `capacity`, checked against
/// SM120 when the op is prepared.
pub(super) fn plans(
    capacity: u32,
    plan: impl Fn(u32) -> Result<LaunchPlan, Refusal>,
) -> Result<Vec<LaunchPlan>, Refusal> {
    (1..=capacity)
        .map(|rows| {
            let plan = plan(rows)?;
            check_plan(&plan)?;
            Ok(plan)
        })
        .collect()
}

/// The prepared plan of `counts.rows`.
pub(super) fn planned(plans: &[LaunchPlan], counts: Counts) -> Result<LaunchPlan, Refusal> {
    counts
        .rows
        .checked_sub(1)
        .and_then(|index| plans.get(index as usize))
        .copied()
        .ok_or(Refusal("Sinkhorn rows outside the prepared capacity"))
}

pub(super) fn dense(rows: u32, elements: u64, bytes: u64) -> Result<u64, Refusal> {
    u64::from(rows)
        .checked_mul(elements)
        .and_then(|n| n.checked_mul(bytes))
        .ok_or(Refusal("Sinkhorn buffer bytes overflow"))
}

pub(super) fn buffer(view: View) -> Argument {
    Argument::DeviceBuffer(view.handle)
}

/// Checks a mixer's weights.
pub(super) fn check_mixer(mix: SinkhornMix, weights: &MixerWeights<View>) -> Result<(), Refusal> {
    let fn_bytes = if mix.fn_dtype == DType::F32 { 4 } else { 2 };
    let mixes = u64::from(mix.mixes());
    weights
        .fn_
        .covers(dense(mix.mixes(), u64::from(mix.flat_width()), fn_bytes)?)?;
    weights.base.covers(mixes * 4)?;
    weights.scale.covers(3 * 4)
}

/// Checks the rows a mix-collapse writes.
fn check_mix_rows(
    mix: SinkhornMix,
    rows: u32,
    collapsed: View,
    post: View,
    comb: View,
) -> Result<(), Refusal> {
    let streams = u64::from(mix.streams);
    collapsed.covers(dense(rows, u64::from(mix.width), 2)?)?;
    post.covers(dense(rows, streams, 2)?)?;
    comb.covers(dense(rows, streams * streams, 2)?)
}

/// The split kernels' chain sums and tickets for `rows` rows.
fn split_scratch(
    mix: SinkhornMix,
    rows: u32,
    scratch: &MixScratch<View>,
) -> Result<(View, View), Refusal> {
    let (Some(partials), Some(tickets)) = (scratch.partials, scratch.tickets) else {
        return Err(Refusal(
            "the split Sinkhorn kernels take their chain sums and tickets",
        ));
    };
    partials.covers(split::partials_bytes(mix, rows))?;
    tickets.covers(split::tickets_bytes(rows))?;
    Ok((partials, tickets))
}

/// The split kernels' scratch for up to `capacity` rows, by role.
fn split_bytes(mix: SinkhornMix, capacity: u32) -> Vec<(String, u64)> {
    vec![
        (
            "scratch.partials".into(),
            split::partials_bytes(mix, capacity),
        ),
        ("scratch.tickets".into(), split::tickets_bytes(capacity)),
    ]
}

/// What a one-CTA entry launches: its symbol, ABI and the code its
/// query's shape and kernel compile; its capacity only bounds the rows.
fn one_cta_launch(entry: &sinkhorn_hc::Entry) -> String {
    let sinkhorn_hc::Entry {
        profile: _, // filters catalogs; no launch reads it
        symbol,     // the launched symbol
        query,      // its shape selects the op and is the code's
        op,         // the kernel; selects the op
        params,     // the ABI
    } = entry;
    format!("{symbol} {:?} {op:?} {params:?}", uncapped(*query))
}

/// What a split mix-collapse entry launches, as `one_cta_launch`.
fn split_launch(entry: &split::Entry) -> String {
    let split::Entry {
        profile: _, // filters catalogs; no launch reads it
        symbol,     // the launched symbol
        query,      // its shape selects the op and is the code's
        params,     // the ABI
    } = entry;
    format!("{symbol} {:?} {params:?}", uncapped(*query))
}

/// What a tiled mix-collapse entry launches, as `one_cta_launch`, with
/// its tile of rows per CTA.
fn tiled_launch(entry: &tiled::Entry) -> String {
    let tiled::Entry {
        profile: _, // filters catalogs; no launch reads it
        symbol,     // the launched symbol
        query,      // its shape selects the op and is the code's
        rows,       // the code and the grid; selects the row counts
        probe,      // the code and the ABI; a probe serves no op
        params,     // the ABI
    } = entry;
    format!("{symbol} {:?} {rows} {probe} {params:?}", uncapped(*query))
}

/// What a fused mix-collapse and norm entry launches, as `one_cta_launch`,
/// with its norm.
fn mix_norm_launch(entry: &mix_norm_tree::Entry) -> String {
    let mix_norm_tree::Entry {
        profile: _, // filters catalogs; no launch reads it
        symbol,     // the launched symbol
        query,      // its shape selects the op and is the code's
        norm,       // the code and the eps argument; selects the op
        params,     // the ABI
    } = entry;
    format!("{symbol} {:?} {norm:?} {params:?}", uncapped(*query))
}

/// What a comm epilogue entry launches: its symbol, mixer, the norm
/// epsilon it passes, its norm's order and its ABI; its capacity only
/// bounds the rows.
fn epilogue_launch(entry: &sinkhorn_split::Entry) -> String {
    let sinkhorn_split::Entry {
        symbol,           // the launched symbol
        mix,              // the code; selects the op
        rmsnorm_eps_bits, // an argument; selects the op
        norm,             // the code's sum of squares; selects the op
        capacity: _,      // bounds the rows the selection settles
        params,           // the ABI
    } = entry;
    format!("{symbol} {mix:?} {rmsnorm_eps_bits} {norm:?} {params:?}")
}

/// The one-CTA entries that run `op` as `kind`.
fn one_cta(selection: &Selection, op: Query, kind: Op) -> Vec<Candidate<sinkhorn_hc::Entry>> {
    selection
        .entries
        .iter()
        .filter(|entry| entry.op == kind && uncapped(entry.query) == uncapped(op))
        .map(|entry| {
            let capacity = entry.contract().and_then(|_| entry.capacity());
            Candidate::new(
                entry.clone(),
                &entry.symbol,
                one_cta_launch(entry),
                capacity,
            )
            .claims(entry.invariance())
        })
        .collect()
}

/// The entry that runs a mix-collapse at a row count.
enum MixKernel {
    OneCta(sinkhorn_hc::Entry),
    Split(split::Entry),
    Tiled(tiled::Entry),
}

impl MixKernel {
    fn symbol(&self) -> &str {
        match self {
            MixKernel::OneCta(entry) => &entry.symbol,
            MixKernel::Split(entry) => &entry.symbol,
            MixKernel::Tiled(entry) => &entry.symbol,
        }
    }

    fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        match self {
            MixKernel::OneCta(entry) => entry.plan(rows, rows),
            MixKernel::Split(entry) => entry.plan(rows, rows),
            MixKernel::Tiled(entry) => entry.plan(rows, rows),
        }
    }

    /// The entry's declaration, whose rows claim refers to the one-CTA
    /// mix-collapse's bits for every form.
    fn invariance(&self) -> Invariance {
        match self {
            MixKernel::OneCta(entry) => entry.invariance(),
            MixKernel::Split(entry) => entry.invariance(),
            MixKernel::Tiled(entry) => entry.invariance(),
        }
    }
}

/// What a mix-collapse claims that launches `kernels[at]` at each row
/// count of `plans` (from one): the routes' claims over the one-CTA
/// reference every form's declaration names (`Invariance::routed`), and
/// any chunk split when every route declares it and the rows claim covers
/// every row count, since each row then has its one-row bits in every
/// launch and no entry hands state to a later one.
fn mix_guarantees(kernels: &[MixKernel], plans: &[(usize, LaunchPlan)]) -> Invariance {
    let routes = plans
        .iter()
        .zip(1..)
        .map(|(&(at, _), rows)| kernel_core::properties::RowRoute {
            rows,
            invariance: kernels[at].invariance(),
            reference: "the one-CTA mix-collapse",
        })
        .collect::<Vec<_>>();
    let routed = Invariance::routed(&routes);
    let stateless = routes
        .iter()
        .all(|route| route.invariance.chunk == kernel_core::properties::Chunk::Any);
    if stateless && routed.one_row_at(routes.len() as u32) {
        routed.with_chunk(kernel_core::properties::Chunk::Any)
    } else {
        routed
    }
}

/// The tile of rows per CTA SM120's schedule runs a mix-collapse of `rows`
/// rows on, or `None` for the split (or one-CTA) entry:
/// `sinkhorn_hc::tiled::served_prefill_mix`, from kernel-lab timing of the
/// served mixer on the served GPU, the split below 144 rows, the 4-row tile
/// to 1504 rows and the 8-row tile from 1505.
fn scheduled_tile(rows: u32) -> Option<u32> {
    match tiled::served_prefill_mix(rows) {
        tiled::PrefillMix::Split => None,
        tiled::PrefillMix::Tiled(tile) => Some(tile),
    }
}

/// The tiled entry of `tile` rows per CTA that serves `op`, if one does.
fn tiled_entry(
    selection: &Selection,
    op: Query,
    tile: u32,
    need: Need,
) -> Result<Option<tiled::Entry>, Refused> {
    let candidates = selection
        .tiled
        .iter()
        .filter(|entry| entry.rows == tile && !entry.probe && uncapped(entry.query) == uncapped(op))
        .map(|entry| {
            let capacity = entry.contract().and_then(|_| entry.capacity());
            Candidate::new(entry.clone(), &entry.symbol, tiled_launch(entry), capacity)
                .claims(entry.invariance())
        })
        .collect();
    Ok(match fitting(candidates, need, "tiled mix-collapse")? {
        Fitting::Chosen(entry) => Some(entry),
        Fitting::Unserved(_) => None,
    })
}

/// A mix-collapse through the model's split entry when one serves the op,
/// else its one-CTA entry, and through its tiled entry of the scheduled
/// tile at the row counts the schedule tiles, when one serves the op.
struct MixCollapseOp {
    /// The entries the op runs: the split or one-CTA entry first.
    kernels: Vec<MixKernel>,
    mix: SinkhornMix,
    /// Each row count's entry (an index into `kernels`) and plan.
    plans: Vec<(usize, LaunchPlan)>,
    info: OpInfo,
}

pub(super) fn mix_collapse(
    selection: &Selection,
    op: &Query,
    need: Need,
) -> Result<Prepared<MixCollapseBuffers<View>>, Refused> {
    sinkhorn_hc::limits(*op)?;
    let Query::MixCollapse { mix, .. } = *op else {
        return Err(Refused::Rule("expected a Sinkhorn mix-collapse".into()));
    };
    check_need(*op, need)?;
    // The split entry when one serves the op: it writes the one-CTA
    // entry's bits.
    let split = selection
        .split
        .iter()
        .filter(|entry| uncapped(entry.query) == uncapped(*op))
        .map(|entry| {
            let capacity = entry.contract().and_then(|_| entry.capacity());
            Candidate::new(entry.clone(), &entry.symbol, split_launch(entry), capacity)
                .claims(entry.invariance())
        })
        .collect();
    let untiled = match fitting(split, need, "split mix-collapse")? {
        Fitting::Chosen(entry) => MixKernel::Split(entry),
        Fitting::Unserved(split) => {
            let one_cta = one_cta(selection, *op, Op::MixCollapse { probe: false });
            match fitting(one_cta, need, "mix-collapse")? {
                Fitting::Chosen(entry) => MixKernel::OneCta(entry),
                Fitting::Unserved(one_cta) => {
                    return Err(unserved(one_cta.or(split), "mix-collapse"));
                }
            }
        }
    };
    // Each tile the schedule gives within the capacity, and the tiled
    // entry that runs it, if one serves the op.
    let mut kernels = vec![untiled];
    let mut tiles = Vec::new();
    for rows in 1..=need.capacity {
        if let Some(tile) = scheduled_tile(rows)
            && !tiles.iter().any(|&(t, _)| t == tile)
        {
            let at = match tiled_entry(selection, *op, tile, need)? {
                Some(entry) => {
                    kernels.push(MixKernel::Tiled(entry));
                    kernels.len() - 1
                }
                None => 0,
            };
            tiles.push((tile, at));
        }
    }
    let kernel_of = |rows: u32| {
        scheduled_tile(rows)
            .and_then(|tile| tiles.iter().find(|&&(t, _)| t == tile))
            .map_or(0, |&(_, at)| at)
    };
    let plans = (1..=need.capacity)
        .map(|rows| {
            let at = kernel_of(rows);
            let plan = kernels[at].plan(rows)?;
            check_plan(&plan)?;
            Ok((at, plan))
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    let symbols = kernels
        .iter()
        .map(|kernel| (kernel.symbol().to_owned(), 0))
        .collect();
    let mut info = OpInfo::new(MIX_COLLAPSE, mix_guarantees(&kernels, &plans), symbols);
    // The split's scratch for the most rows it runs.
    if let MixKernel::Split(_) = kernels[0]
        && let Some(rows) = (1..=need.capacity).rev().find(|&rows| kernel_of(rows) == 0)
    {
        info.bytes = split_bytes(mix, rows);
    }
    need.check(&info)?;
    Ok(Box::new(MixCollapseOp {
        kernels,
        mix,
        plans,
        info,
    }))
}

impl PreparedOp<MixCollapseBuffers<View>> for MixCollapseOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &MixCollapseBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (at, plan) = counts
            .rows
            .checked_sub(1)
            .and_then(|index| self.plans.get(index as usize))
            .copied()
            .ok_or(Refusal("Sinkhorn rows outside the prepared capacity"))?;
        let rows = counts.rows;
        let mix = self.mix;
        buffers
            .x
            .covers(dense(rows, u64::from(mix.flat_width()), 2)?)?;
        check_mixer(mix, &buffers.mixer)?;
        check_mix_rows(mix, rows, buffers.collapsed, buffers.post, buffers.comb)?;
        buffers.live_rows.covers(4)?;
        let mut values = [
            buffers.x,
            buffers.mixer.fn_,
            buffers.mixer.base,
            buffers.mixer.scale,
            buffers.collapsed,
            buffers.post,
            buffers.comb,
        ]
        .map(buffer)
        .to_vec();
        let kernel = &self.kernels[at];
        let arguments = match kernel {
            MixKernel::Split(entry) => {
                let (partials, tickets) = split_scratch(mix, rows, &buffers.scratch)?;
                values.extend([partials, tickets, buffers.live_rows].map(buffer));
                entry.pack_arguments(values)?
            }
            MixKernel::OneCta(entry) => {
                values.push(buffer(buffers.live_rows));
                entry.pack_arguments(values)?
            }
            MixKernel::Tiled(entry) => {
                values.push(buffer(buffers.live_rows));
                entry.pack_arguments(values)?
            }
        };
        Ok(vec![Launch {
            symbol: kernel.symbol().to_owned(),
            rows,
            plan,
            arguments,
            collective: false,
            persistent: false,
        }])
    }
}

/// The least input-norm eps the fused mix-collapse and norm and the comm
/// epilogue serve, 2^-101 (biased exponent 26).
///
/// The contract rounds the mean of squares twice, `fl(fl(s / H) + eps)`.
/// The tree entry's `total / Group + eps` is what nvcc makes of it: at the
/// power-of-two widths it takes (1024, 2048, 4096) one `fma.rn.f32(s, 2^-k,
/// eps)`, rounded once; at 3072 `div.rn.f32` then `add.f32`, the contract's
/// two roundings (PTX of every width the entry builds, `-O3`). The forms
/// differ only where `s / H` is not exact, below 2^-126: from eps = 2^-101
/// the sum's ulp is at least 2^-124, so that quotient, rounded or not, is
/// under half an ulp and both give eps, while a normal quotient is exact.
/// Smaller epsilons are refused. The comm epilogue's norm at the fused
/// width 4096 compiles the same FMA (its split tree entry,
/// `fma.rn.f32(s, 2^-12, eps)`).
const MIN_NORM_EPS_BITS: u32 = 26 << 23;

/// A mix-collapse and its input norm through the model's split entry fused
/// with the tree norm.
struct MixCollapseNormOp {
    entry: mix_norm_tree::Entry,
    mix: SinkhornMix,
    plans: Vec<LaunchPlan>,
    info: OpInfo,
}

pub(super) fn mix_collapse_norm(
    selection: &Selection,
    op: &Query,
    need: Need,
) -> Result<Prepared<MixCollapseNormBuffers<View>>, Refused> {
    sinkhorn_hc::limits(*op)?;
    let Query::MixCollapseNorm {
        mix,
        rmsnorm_eps_bits,
        ..
    } = *op
    else {
        return Err(Refused::Rule(
            "expected a Sinkhorn mix-collapse and norm".into(),
        ));
    };
    check_need(*op, need)?;
    // Positive finite epsilons order as their bits.
    if rmsnorm_eps_bits < MIN_NORM_EPS_BITS {
        return Err(Refused::Rule(
            "SM120's fused Sinkhorn mix and norm rounds its mean of squares with eps once; it serves norm epsilons from 2^-101, where that equals the contract's two roundings".into(),
        ));
    }
    // The entry's mix-collapse and its norm: the HF kind with the tree's
    // numerics over the collapsed row at the query's eps.
    let collapse = Query::MixCollapse { mix, capacity: 0 };
    let norm = mix_norm_tree::norm_requirement(mix, f32::from_bits(rmsnorm_eps_bits));
    let candidates = selection
        .mix_norm
        .iter()
        .filter(|entry| uncapped(entry.query) == collapse && entry.norm == norm)
        .map(|entry| {
            let capacity = entry.contract().and_then(|_| entry.capacity());
            Candidate::new(
                entry.clone(),
                &entry.symbol,
                mix_norm_launch(entry),
                capacity,
            )
            .claims(entry.invariance())
        })
        .collect();
    let name = "mix-collapse and norm";
    let entry = chosen(fitting(candidates, need, name)?, name)?;
    let plans = plans(need.capacity, |rows| entry.plan(rows, rows))?;
    let mut info = OpInfo::new(
        MIX_COLLAPSE_NORM,
        entry.invariance(),
        vec![(entry.symbol.clone(), 0)],
    );
    info.bytes = split_bytes(mix, need.capacity);
    need.check(&info)?;
    Ok(Box::new(MixCollapseNormOp {
        entry,
        mix,
        plans,
        info,
    }))
}

impl PreparedOp<MixCollapseNormBuffers<View>> for MixCollapseNormOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &MixCollapseNormBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let plan = planned(&self.plans, counts)?;
        let rows = counts.rows;
        let mix = self.mix;
        buffers
            .x
            .covers(dense(rows, u64::from(mix.flat_width()), 2)?)?;
        check_mixer(mix, &buffers.mixer)?;
        buffers.norm_weight.covers(u64::from(mix.width) * 2)?;
        check_mix_rows(mix, rows, buffers.collapsed, buffers.post, buffers.comb)?;
        buffers
            .normed
            .covers(dense(rows, u64::from(mix.width), 2)?)?;
        buffers.live_rows.covers(4)?;
        let (partials, tickets) = split_scratch(mix, rows, &buffers.scratch)?;
        let mut values = [
            buffers.x,
            buffers.mixer.fn_,
            buffers.mixer.base,
            buffers.mixer.scale,
            buffers.norm_weight,
            buffers.collapsed,
            buffers.post,
            buffers.comb,
            buffers.normed,
            partials,
            tickets,
            buffers.live_rows,
        ]
        .map(buffer)
        .to_vec();
        values.push(Argument::U32(self.entry.norm.epsilon_bits));
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows,
            plan,
            arguments: self.entry.pack_arguments(values)?,
            collective: false,
            persistent: false,
        }])
    }
}

/// A one-CTA stream entry (expand or head) of buffers `K`.
struct StreamOp<K> {
    entry: sinkhorn_hc::Entry,
    streams: u64,
    width: u64,
    plans: Vec<LaunchPlan>,
    info: OpInfo,
    kind: std::marker::PhantomData<fn() -> K>,
}

impl<K> StreamOp<K> {
    fn new(
        selection: &Selection,
        op: Query,
        kind: Op,
        implementation: &str,
        need: Need,
    ) -> Result<Self, Refused> {
        sinkhorn_hc::limits(op)?;
        let (streams, width) = match (op, kind) {
            (Query::Expand { streams, width, .. }, Op::Expand)
            | (Query::Head { streams, width, .. }, Op::Head) => (streams, width),
            _ => {
                return Err(Refused::Rule(format!("expected a Sinkhorn {kind:?}")));
            }
        };
        check_need(op, need)?;
        let name = format!("{kind:?}");
        let entry = chosen(fitting(one_cta(selection, op, kind), need, &name)?, &name)?;
        let plans = plans(need.capacity, |rows| entry.plan(rows, rows))?;
        let info = OpInfo::new(
            implementation,
            entry.invariance(),
            vec![(entry.symbol.clone(), 0)],
        );
        need.check(&info)?;
        Ok(Self {
            entry,
            streams: u64::from(streams),
            width: u64::from(width),
            plans,
            info,
            kind: std::marker::PhantomData,
        })
    }

    fn launch(&self, rows: u32, plan: LaunchPlan, values: &[View]) -> Result<Vec<Launch>, Refusal> {
        Ok(vec![Launch {
            symbol: self.entry.symbol.clone(),
            rows,
            plan,
            arguments: self
                .entry
                .pack_arguments(values.iter().copied().map(buffer).collect())?,
            collective: false,
            persistent: false,
        }])
    }
}

pub(super) fn expand(
    selection: &Selection,
    op: &Query,
    need: Need,
) -> Result<Prepared<StreamExpandBuffers<View>>, Refused> {
    let op = StreamOp::new(selection, *op, Op::Expand, "sm120.sinkhorn.expand.v1", need)?;
    Ok(Box::new(op))
}

impl PreparedOp<StreamExpandBuffers<View>> for StreamOp<StreamExpandBuffers<View>> {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &StreamExpandBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let plan = planned(&self.plans, counts)?;
        let rows = counts.rows;
        let wide = dense(rows, self.streams * self.width, 2)?;
        buffers.x.covers(wide)?;
        buffers.y.covers(dense(rows, self.width, 2)?)?;
        buffers.post.covers(dense(rows, self.streams, 2)?)?;
        buffers
            .comb
            .covers(dense(rows, self.streams * self.streams, 2)?)?;
        buffers.output.covers(wide)?;
        buffers.live_rows.covers(4)?;
        self.launch(
            rows,
            plan,
            &[
                buffers.x,
                buffers.y,
                buffers.post,
                buffers.comb,
                buffers.output,
                buffers.live_rows,
            ],
        )
    }
}

pub(super) fn head(
    selection: &Selection,
    op: &Query,
    need: Need,
) -> Result<Prepared<StreamHeadBuffers<View>>, Refused> {
    let op = StreamOp::new(selection, *op, Op::Head, "sm120.sinkhorn.head.v1", need)?;
    Ok(Box::new(op))
}

impl PreparedOp<StreamHeadBuffers<View>> for StreamOp<StreamHeadBuffers<View>> {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &StreamHeadBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let plan = planned(&self.plans, counts)?;
        let rows = counts.rows;
        buffers
            .x
            .covers(dense(rows, self.streams * self.width, 2)?)?;
        buffers.output.covers(dense(rows, self.width, 2)?)?;
        buffers.live_rows.covers(4)?;
        self.launch(rows, plan, &[buffers.x, buffers.output, buffers.live_rows])
    }
}

/// The comm epilogue on one rank: the all-reduce of the sublayer's
/// partials (its collective launches and per-rank state), then the model's
/// split epilogue over the rank's F32 sums, which waits for no peer. The
/// all-reduce takes the model's registrations as `Backend::all_reduce`
/// does: the kernel base's F32 all-reduce into the sums, or the
/// BF16-partials all-reduce into the rounded sums, which the model's widen
/// row then writes into the F32 sum region the op adds to its state.
struct ReduceExpandMixNormOp {
    reduce: Reduce,
    entry: sinkhorn_split::Entry,
    mix: SinkhornMix,
    plans: Vec<LaunchPlan>,
    info: OpInfo,
}

/// The comm epilogue's all-reduce.
enum Reduce {
    F32(Prepared<AllReduceBuffers<View>>),
    Bf16(Box<super::bf16_all_reduce::Bf16AllReduceOp>),
}

pub(super) fn reduce_expand_mix_norm(
    selection: &Selection,
    low_latency: &[super::all_reduce::LowLatency],
    bf16: &super::bf16_all_reduce::Selection,
    op: &Query,
    rank: u32,
    need: Need,
) -> Result<Prepared<ReduceExpandMixNormBuffers<View>>, Refused> {
    sinkhorn_hc::limits(*op)?;
    let Query::AllReduceExpandMixNorm {
        ranks,
        dtype,
        mix,
        rmsnorm_eps_bits,
        norm,
        ..
    } = *op
    else {
        return Err(Refused::Rule(
            "expected a Sinkhorn all-reduce, expand, mix and norm".into(),
        ));
    };
    check_need(*op, need)?;
    if rmsnorm_eps_bits < MIN_NORM_EPS_BITS {
        return Err(Refused::Rule(
            "SM120's Sinkhorn comm epilogue rounds its mean of squares with eps once; it serves norm epsilons from 2^-101, where that equals the contract's two roundings".into(),
        ));
    }
    // Each entry's contract applies SM120's limits: the fused width, four
    // streams and the split's shared memory. An entry serves only the norm
    // order its kernel sums in.
    let candidates = selection
        .epilogues
        .iter()
        .filter(|entry| {
            entry.mix == mix && entry.rmsnorm_eps_bits == rmsnorm_eps_bits && entry.norm == norm
        })
        .map(|entry| {
            let capacity = entry.contract().map(|_| entry.capacity);
            Candidate::new(
                entry.clone(),
                &entry.symbol,
                epilogue_launch(entry),
                capacity,
            )
            .claims(entry.invariance())
        })
        .collect();
    let entry = chosen(fitting(candidates, need, "comm epilogue")?, "comm epilogue")?;
    let sum = CommQuery::AllReduce {
        ranks,
        elements: u64::from(mix.width),
        dtype,
        fused_residual: false,
        fused_rmsnorm: None,
    };
    let rows = Need::rows(need.capacity);
    let reduce = match dtype {
        DType::Bf16 => Reduce::Bf16(Box::new(super::bf16_all_reduce::Bf16AllReduceOp::new(
            bf16, sum, rank, rows, true,
        )?)),
        _ => Reduce::F32(super::all_reduce::prepare(low_latency, &sum, rank, rows)?),
    };
    let plans = plans(need.capacity, |rows| entry.plan(rows, ranks))?;
    let (summed, widen) = match &reduce {
        Reduce::F32(reduce) => (reduce.info(), None),
        Reduce::Bf16(reduce) => (reduce.info(), Some(reduce.widen_claims())),
    };
    let mut symbols = summed.symbols.clone();
    symbols.push((entry.symbol.clone(), 0));
    let guarantees = [summed.guarantees, entry.invariance()]
        .into_iter()
        .chain(widen)
        .reduce(Invariance::meet)
        .unwrap_or(Invariance::NONE);
    let mut info = OpInfo::new(
        format!("{}+{EXPAND_MIX_NORM}", summed.implementation),
        guarantees,
        symbols,
    );
    info.state = summed.state.clone();
    info.workspace = summed.workspace;
    info.bytes = split_bytes(mix, need.capacity);
    need.check(&info)?;
    Ok(Box::new(ReduceExpandMixNormOp {
        reduce,
        entry,
        mix,
        plans,
        info,
    }))
}

impl PreparedOp<ReduceExpandMixNormBuffers<View>> for ReduceExpandMixNormOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &ReduceExpandMixNormBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let plan = planned(&self.plans, counts)?;
        let rows = counts.rows;
        let mix = self.mix;
        let wide = dense(rows, u64::from(mix.flat_width()), 2)?;
        let narrow = dense(rows, u64::from(mix.width), 2)?;
        // The all-reduce's workspace is the op's; its F32 sums are the
        // epilogue's input.
        let (mut launches, summed) = match &self.reduce {
            Reduce::F32(reduce) => (
                reduce.launches(&buffers.reduce, workspace, counts)?,
                buffers.reduce.output,
            ),
            Reduce::Bf16(reduce) => {
                let mut launches = reduce.launches(&buffers.reduce, workspace, counts)?;
                launches.push(reduce.widen_sum(&buffers.reduce, rows)?);
                (launches, reduce.sum_view(&buffers.reduce)?)
            }
        };
        buffers.streams.covers(wide)?;
        buffers.reduced.covers(narrow)?;
        buffers.expanded.covers(wide)?;
        check_mixer(mix, &buffers.mixer)?;
        check_mix_rows(mix, rows, buffers.collapsed, buffers.post, buffers.comb)?;
        buffers.norm_weight.covers(u64::from(mix.width) * 2)?;
        buffers.normed.covers(narrow)?;
        buffers.live_rows.covers(4)?;
        let (partials, tickets) = split_scratch(mix, rows, &buffers.scratch)?;
        let epilogue = EpilogueBuffers {
            streams: buffers.streams.handle,
            post: buffers.post.handle,
            comb: buffers.comb.handle,
            reduced: buffers.reduced.handle,
            expanded: buffers.expanded.handle,
            fn_: buffers.mixer.fn_.handle,
            base: buffers.mixer.base.handle,
            scale: buffers.mixer.scale.handle,
            collapsed: buffers.collapsed.handle,
            weight: buffers.norm_weight.handle,
            normed: buffers.normed.handle,
            live_rows: buffers.live_rows.handle,
            eps_bits: self.entry.rmsnorm_eps_bits,
        };
        let split = SplitBuffers {
            epilogue,
            partials: partials.handle,
            tickets: tickets.handle,
        };
        launches.push(Launch {
            symbol: self.entry.symbol.clone(),
            rows,
            plan,
            arguments: sinkhorn_split::rows_arguments(summed.handle, split)?,
            collective: false,
            persistent: false,
        });
        Ok(launches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use crate::sinkhorn_hc::{Geometry, SERVED};
    use crate::{Profile, ProfileNames};
    use kernel_build::EntryParam;
    use kernel_core::backend::Backend;
    use kernel_core::collective::RankViews;
    use kernel_core::contract::Validate;
    use kernel_core::properties::Ties;
    use kernel_core::rmsnorm::RmsNormReduction;
    use kernel_core::sinkhorn_hyper_connection::StreamHeadMode;

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "sinkhorn-backend-test",
        tag: "test",
    });

    fn eps() -> u32 {
        1e-5_f32.to_bits()
    }

    /// Four streams of 4096 up to 8192 rows, the shape the Sinkhorn
    /// entries serve: the one-CTA entries with the lab's probe, and the
    /// split mix-collapse and the comm epilogue when asked.
    fn backend(split: bool, epilogue: bool) -> Sm120 {
        let mut backend = Sm120::new().with_sinkhorn(sinkhorn_hc::entries(PROFILE, SERVED, true));
        if split {
            backend = backend.with_sinkhorn_split(split::entry(PROFILE, SERVED));
        }
        if epilogue {
            backend = backend.with_sinkhorn_epilogue(
                sinkhorn_split::entry(PROFILE, SERVED.mix, eps(), SERVED.capacity).unwrap(),
            );
        }
        backend
    }

    fn mix_collapse(capacity: u32) -> Query {
        Query::MixCollapse {
            mix: SERVED.mix,
            capacity,
        }
    }

    fn mixer(first: u64) -> MixerWeights<View> {
        MixerWeights {
            fn_: view(first, u64::MAX),
            base: view(first + 1, u64::MAX),
            scale: view(first + 2, u64::MAX),
        }
    }

    fn scratch(first: u64) -> MixScratch<View> {
        MixScratch {
            partials: Some(view(first, u64::MAX)),
            tickets: Some(view(first + 1, u64::MAX)),
        }
    }

    /// Handles 1 to 10 in the split entry's parameter order.
    fn mix_buffers() -> MixCollapseBuffers<View> {
        MixCollapseBuffers {
            x: view(1, u64::MAX),
            mixer: mixer(2),
            collapsed: view(5, u64::MAX),
            post: view(6, u64::MAX),
            comb: view(7, u64::MAX),
            scratch: scratch(8),
            live_rows: view(10, 4),
        }
    }

    fn handles(handles: impl IntoIterator<Item = u64>) -> Vec<Argument> {
        handles.into_iter().map(Argument::DeviceBuffer).collect()
    }

    /// Every served entry declares one-row bits up to its capacity, any
    /// chunk split and row independence; the probes declare nothing.
    #[test]
    fn served_sinkhorn_entries_declare_any_rows() {
        let served = Invariance::any_rows(8192);
        for entry in sinkhorn_hc::entries(PROFILE, SERVED, true) {
            let expected = match entry.op {
                Op::MixCollapse { probe: true } => Invariance::NONE,
                _ => served,
            };
            assert_eq!(entry.invariance(), expected, "{}", entry.symbol);
        }
        assert_eq!(split::entry(PROFILE, SERVED).invariance(), served);
        for entry in tiled::lab_entries() {
            let expected = if entry.probe {
                Invariance::NONE
            } else {
                served
            };
            assert_eq!(entry.invariance(), expected, "{}", entry.symbol);
        }
        assert_eq!(mix_norm_tree::served().invariance(), served);
        let narrow = Geometry {
            capacity: 32,
            ..SERVED
        };
        assert_eq!(
            split::entry(PROFILE, narrow).invariance(),
            Invariance::any_rows(32)
        );
    }

    /// The mix-collapse claims one-row bits and any chunk split across
    /// its forms (split, then the 4- and 8-row tiles from 144 and 1505
    /// rows), since each declares the one-CTA bits at every row count; a
    /// form declaring nothing leaves the op no claim.
    #[test]
    fn mix_collapse_claims_one_row_bits_across_its_forms() {
        let backend = backend(true, false).with_sinkhorn_tiled(
            tiled::SERVED_TILES.map(|rows| tiled::entry(PROFILE, SERVED, rows)),
        );
        let prepared = backend
            .sinkhorn_mix_collapse(
                &mix_collapse(8192),
                Need::rows(8192).with_exact(Invariance::any_rows(8192)),
            )
            .unwrap();
        assert_eq!(prepared.info().symbols.len(), 3);
        assert_eq!(prepared.info().guarantees, Invariance::any_rows(8192));
        let kernels = vec![
            MixKernel::Split(split::entry(PROFILE, SERVED)),
            MixKernel::Tiled(tiled::entry_with_suffix(PROFILE, SERVED, 4, true, "")),
        ];
        let plan = kernels[0].plan(1).unwrap();
        let plans = (1..=200)
            .map(|rows| (usize::from(rows >= 144), plan))
            .collect::<Vec<_>>();
        assert_eq!(mix_guarantees(&kernels, &plans), Invariance::NONE);
        assert_eq!(
            mix_guarantees(&kernels, &plans[..143]),
            Invariance::any_rows(8192)
        );
    }

    #[test]
    fn mix_collapse_launches_the_split_entry_as_its_builder_does() {
        let entry = split::entry(PROFILE, SERVED);
        let prepared = backend(true, true)
            .sinkhorn_mix_collapse(&mix_collapse(8192), Need::rows(8192))
            .unwrap();
        let info = prepared.info();
        assert_eq!(info.implementation, MIX_COLLAPSE);
        assert_eq!(info.guarantees, Invariance::any_rows(8192));
        assert_eq!(info.symbols, [(entry.symbol.clone(), 0)]);
        assert_eq!(
            info.bytes_of("scratch.partials"),
            Some(split::partials_bytes(SERVED.mix, 8192))
        );
        assert_eq!(info.bytes_of("scratch.tickets"), Some(8192 * 4));
        assert!(info.weight_layouts.is_empty());
        // Decode batches, verify buckets of six rows per sequence and
        // prefill chunks.
        for rows in [1, 2, 32, 6 * 32, 2048, 8192] {
            let launches = prepared
                .launches(&mix_buffers(), View::EMPTY, Counts::rows(rows))
                .unwrap();
            assert_eq!(
                launches,
                [Launch {
                    symbol: entry.symbol.clone(),
                    rows,
                    plan: entry.plan(rows, rows).unwrap(),
                    arguments: entry.pack_arguments(handles(1..=10)).unwrap(),
                    collective: false,
                    persistent: false,
                }]
            );
        }
    }

    #[test]
    fn mix_collapse_without_a_split_entry_launches_the_one_cta_entry() {
        let entry = sinkhorn_hc::entries(PROFILE, SERVED, false).remove(0);
        assert_eq!(entry.op, Op::MixCollapse { probe: false });
        let prepared = backend(false, false)
            .sinkhorn_mix_collapse(&mix_collapse(256), Need::rows(256))
            .unwrap();
        // The same numerics as the split entry, and no scratch.
        assert_eq!(prepared.info().implementation, MIX_COLLAPSE);
        assert!(prepared.info().bytes.is_empty());
        let buffers = MixCollapseBuffers {
            scratch: MixScratch {
                partials: None,
                tickets: None,
            },
            ..mix_buffers()
        };
        for rows in [1, 32, 256] {
            let launches = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            assert_eq!(
                launches,
                [Launch {
                    symbol: entry.symbol.clone(),
                    rows,
                    plan: entry.plan(rows, rows).unwrap(),
                    arguments: entry
                        .pack_arguments(handles([1, 2, 3, 4, 5, 6, 7, 10]))
                        .unwrap(),
                    collective: false,
                    persistent: false,
                }]
            );
        }
    }

    /// SM120's tile schedule over a mix-collapse's row counts, stated as a
    /// table: (first rows, last rows, tile of rows per CTA or `None` for
    /// the split entry).
    const TILE_TABLE: [(u32, u32, Option<u32>); 3] =
        [(1, 143, None), (144, 1504, Some(4)), (1505, 8192, Some(8))];

    fn tiled_backend(tiles: &[u32], probe: bool) -> Sm120 {
        backend(true, false).with_sinkhorn_tiled(
            tiles
                .iter()
                .map(|&rows| tiled::entry_with_suffix(PROFILE, SERVED, rows, probe, "")),
        )
    }

    /// The launch a mix-collapse of `rows` rows makes on `tile`'s tiled
    /// entry, or on the split entry, over [`mix_buffers`].
    fn mix_launch(tile: Option<u32>, rows: u32) -> Launch {
        let (symbol, plan, arguments) = match tile {
            None => {
                let entry = split::entry(PROFILE, SERVED);
                let arguments = entry.pack_arguments(handles(1..=10)).unwrap();
                (
                    entry.symbol.clone(),
                    entry.plan(rows, rows).unwrap(),
                    arguments,
                )
            }
            Some(tile) => {
                let entry = tiled::entry(PROFILE, SERVED, tile);
                let arguments = entry
                    .pack_arguments(handles([1, 2, 3, 4, 5, 6, 7, 10]))
                    .unwrap();
                (
                    entry.symbol.clone(),
                    entry.plan(rows, rows).unwrap(),
                    arguments,
                )
            }
        };
        Launch {
            symbol,
            rows,
            plan,
            arguments,
            collective: false,
            persistent: false,
        }
    }

    #[test]
    fn the_mix_collapse_runs_the_tile_the_table_states_at_every_row_count() {
        // The 16-row tile too, which the schedule never picks.
        let prepared = tiled_backend(&[16, 8, 4], false)
            .sinkhorn_mix_collapse(&mix_collapse(8192), Need::rows(8192))
            .unwrap();
        let info = prepared.info();
        assert_eq!(info.implementation, MIX_COLLAPSE);
        assert_eq!(
            info.symbols,
            [
                ("chop_sinkhorn_hc_test_mix_collapse_split_v1".to_owned(), 0),
                (
                    "chop_sinkhorn_hc_test_mix_collapse_tiled_r4_v1".to_owned(),
                    0
                ),
                (
                    "chop_sinkhorn_hc_test_mix_collapse_tiled_r8_v1".to_owned(),
                    0
                ),
            ]
        );
        // The split runs 143 rows at most.
        assert_eq!(
            info.bytes,
            [
                (
                    "scratch.partials".to_owned(),
                    split::partials_bytes(SERVED.mix, 143)
                ),
                ("scratch.tickets".to_owned(), split::tickets_bytes(143)),
            ]
        );
        let mut seen = 0;
        for (first, last, tile) in TILE_TABLE {
            for rows in first..=last {
                let launches = prepared
                    .launches(&mix_buffers(), View::EMPTY, Counts::rows(rows))
                    .unwrap();
                assert_eq!(launches, [mix_launch(tile, rows)], "{rows} rows");
                seen += 1;
            }
        }
        assert_eq!(seen, 8192);
        // The tiled entries take no chain sums or tickets.
        let unscratched = MixCollapseBuffers {
            scratch: MixScratch {
                partials: None,
                tickets: None,
            },
            ..mix_buffers()
        };
        for rows in [144, 1505, 8192] {
            prepared
                .launches(&unscratched, View::EMPTY, Counts::rows(rows))
                .unwrap();
        }
        assert!(
            prepared
                .launches(&unscratched, View::EMPTY, Counts::rows(143))
                .is_err()
        );
        // A capacity within a tile lists the entries its rows run.
        for (capacity, symbols, split_rows) in [(100, 1, 100), (1504, 2, 143), (1505, 3, 143)] {
            let prepared = tiled_backend(&[4, 8], false)
                .sinkhorn_mix_collapse(&mix_collapse(8192), Need::rows(capacity))
                .unwrap();
            assert_eq!(prepared.info().symbols.len(), symbols, "{capacity}");
            assert_eq!(
                prepared.info().bytes_of("scratch.partials"),
                Some(split::partials_bytes(SERVED.mix, split_rows))
            );
        }
    }

    #[test]
    fn rows_without_a_serving_tiled_entry_keep_the_untiled_entry() {
        let symbols_at = |backend: Sm120, rows: &[u32]| {
            let prepared = backend
                .sinkhorn_mix_collapse(&mix_collapse(8192), Need::rows(8192))
                .unwrap();
            rows.iter()
                .map(|&rows| {
                    prepared
                        .launches(&mix_buffers(), View::EMPTY, Counts::rows(rows))
                        .unwrap()
                        .remove(0)
                        .symbol
                })
                .collect::<Vec<_>>()
        };
        let split = "chop_sinkhorn_hc_test_mix_collapse_split_v1";
        let r8 = "chop_sinkhorn_hc_test_mix_collapse_tiled_r8_v1";
        // Only the 8-row tile: the 4-row tile's rows run the split.
        assert_eq!(
            symbols_at(tiled_backend(&[8], false), &[143, 144, 1504, 1505]),
            [split, split, split, r8]
        );
        // Probe entries serve no op.
        assert_eq!(
            symbols_at(tiled_backend(&[4, 8], true), &[144, 1505]),
            [split, split]
        );
        // A tiled entry too narrow for the prepared rows serves none.
        let narrow = backend(true, false).with_sinkhorn_tiled([tiled::entry(
            PROFILE,
            Geometry {
                mix: SERVED.mix,
                capacity: 4096,
            },
            8,
        )]);
        assert_eq!(symbols_at(narrow, &[1505, 8192]), [split, split]);
        // Without a split entry the one-CTA entry runs the untiled rows.
        let one_cta = Sm120::new()
            .with_sinkhorn(sinkhorn_hc::entries(PROFILE, SERVED, false))
            .with_sinkhorn_tiled([4, 8].map(|rows| tiled::entry(PROFILE, SERVED, rows)));
        assert_eq!(
            symbols_at(one_cta, &[1, 143, 144]),
            [
                "chop_sinkhorn_hc_test_mix_collapse_v1",
                "chop_sinkhorn_hc_test_mix_collapse_v1",
                "chop_sinkhorn_hc_test_mix_collapse_tiled_r4_v1"
            ]
        );
        // Tiled entries alone leave the untiled rows unserved.
        assert!(
            Sm120::new()
                .with_sinkhorn_tiled([4, 8].map(|rows| tiled::entry(PROFILE, SERVED, rows)))
                .sinkhorn_mix_collapse(&mix_collapse(8192), Need::rows(8192))
                .is_err()
        );
    }

    #[test]
    fn tiled_entries_are_chosen_whatever_the_registration_order() {
        let prepare = |order: &[(Profile, u32)]| -> Prepared32 {
            let backend =
                order
                    .iter()
                    .fold(backend(true, false), |backend, &(profile, capacity)| {
                        let geometry = Geometry {
                            mix: SERVED.mix,
                            capacity: capacity * 8,
                        };
                        backend.with_sinkhorn_tiled([tiled::entry(profile, geometry, 4)])
                    });
            let prepared = backend.sinkhorn_mix_collapse(&mix_collapse(256), Need::rows(256))?;
            let launches = prepared
                .launches(&mix_buffers(), View::EMPTY, Counts::rows(200))
                .unwrap();
            Ok((prepared.info().clone(), launches))
        };
        // Capacities 64, 256 (twice), 512 and the tie's 256.
        assert_eq!(
            symbols(&in_every_order(&registered(false), prepare)),
            ["chop_sinkhorn_hc_least_mix_collapse_tiled_r4_v1"]
        );
        assert_eq!(
            in_every_order(&registered(true), prepare),
            Err(Refused::Rule(
                "two SM120 Sinkhorn tiled mix-collapse entries serve the op, chop_sinkhorn_hc_least_mix_collapse_tiled_r4_v1 and chop_sinkhorn_hc_tied_mix_collapse_tiled_r4_v1".into()
            ))
        );
    }

    fn mix_norm_query(eps: f32, capacity: u32) -> Query {
        Query::MixCollapseNorm {
            mix: SERVED.mix,
            rmsnorm_eps_bits: eps.to_bits(),
            capacity,
        }
    }

    fn mix_norm_entry(profile: Profile, capacity: u32) -> mix_norm_tree::Entry {
        let geometry = Geometry {
            mix: SERVED.mix,
            capacity,
        };
        mix_norm_tree::entry(
            profile,
            geometry,
            mix_norm_tree::norm_requirement(SERVED.mix, 1e-5),
        )
    }

    /// Handles 1 to 12 in the fused entry's parameter order.
    fn mix_norm_buffers() -> MixCollapseNormBuffers<View> {
        MixCollapseNormBuffers {
            x: view(1, u64::MAX),
            mixer: mixer(2),
            norm_weight: view(5, u64::MAX),
            collapsed: view(6, u64::MAX),
            post: view(7, u64::MAX),
            comb: view(8, u64::MAX),
            normed: view(9, u64::MAX),
            scratch: scratch(10),
            live_rows: view(12, 4),
        }
    }

    #[test]
    fn mix_collapse_norm_launches_the_tree_entry_as_its_builder_does() {
        let entry = mix_norm_entry(PROFILE, SERVED.capacity);
        let backend = backend(true, true).with_sinkhorn_mix_norm(entry.clone());
        let prepared = backend
            .sinkhorn_mix_collapse_norm(&mix_norm_query(1e-5, 8192), Need::rows(8192))
            .unwrap();
        let info = prepared.info();
        assert_eq!(info.implementation, MIX_COLLAPSE_NORM);
        assert_eq!(info.guarantees, Invariance::any_rows(8192));
        assert_eq!(info.symbols, [(entry.symbol.clone(), 0)]);
        assert_eq!(info.bytes, split_bytes(SERVED.mix, 8192));
        let mut arguments = handles(1..=12);
        arguments.push(Argument::U32(1e-5_f32.to_bits()));
        // Decode batches, verify buckets of six rows per sequence and the
        // capacity.
        for rows in [1, 2, 16, 32, 6 * 32, 8192] {
            let launches = prepared
                .launches(&mix_norm_buffers(), View::EMPTY, Counts::rows(rows))
                .unwrap();
            assert_eq!(
                launches,
                [Launch {
                    symbol: entry.symbol.clone(),
                    rows,
                    plan: entry.plan(rows, rows).unwrap(),
                    arguments: entry.pack_arguments(arguments.clone()).unwrap(),
                    collective: false,
                    persistent: false,
                }]
            );
        }
        // Every view and count is checked.
        let short = |change: fn(&mut MixCollapseNormBuffers<View>)| {
            let mut buffers = mix_norm_buffers();
            change(&mut buffers);
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(32))
                .is_err()
        };
        assert!(short(|b| b.normed = view(9, 32 * 4096 * 2 - 1)));
        assert!(short(|b| b.norm_weight = view(5, 4096 * 2 - 1)));
        assert!(short(|b| b.collapsed = view(6, 32 * 4096 * 2 - 1)));
        assert!(short(|b| b.scratch.tickets = None));
        assert!(short(|b| b.live_rows = view(12, 3)));
        assert!(!short(|b| b.normed = view(9, 32 * 4096 * 2)));
        for rows in [0, 8193] {
            assert!(
                prepared
                    .launches(&mix_norm_buffers(), View::EMPTY, Counts::rows(rows))
                    .is_err()
            );
        }
    }

    #[test]
    fn mix_collapse_norm_refuses_what_no_tree_entry_serves() {
        let entry = mix_norm_entry(PROFILE, 256);
        let backend = Sm120::new().with_sinkhorn_mix_norm(entry.clone());
        let prepare = |op: Query, need: Need| backend.sinkhorn_mix_collapse_norm(&op, need).err();
        // Another eps, past the entry's rows, an exactness claim, the
        // unfused query and past the SM120 kernels.
        assert_eq!(
            prepare(mix_norm_query(1e-6, 256), Need::rows(32)),
            Some(Refused::Rule(
                "no SM120 Sinkhorn mix-collapse and norm entry serves the query".into()
            ))
        );
        assert!(prepare(mix_norm_query(1e-5, 512), Need::rows(512)).is_some());
        assert!(prepare(mix_norm_query(1e-5, 256), Need::rows(257)).is_some());
        assert!(matches!(
            prepare(
                mix_norm_query(1e-5, 256),
                Need::rows(32).with_exact(Invariance::one_row(512))
            ),
            Some(Refused::Inexact { .. })
        ));
        assert!(
            prepare(
                mix_norm_query(1e-5, 256),
                Need::rows(32).with_exact(Invariance::any_rows(256))
            )
            .is_none()
        );
        assert_eq!(
            prepare(mix_collapse(256), Need::rows(32)),
            Some(Refused::Rule(
                "expected a Sinkhorn mix-collapse and norm".into()
            ))
        );
        let wide = Query::MixCollapseNorm {
            mix: SinkhornMix {
                streams: 9,
                ..SERVED.mix
            },
            rmsnorm_eps_bits: eps(),
            capacity: 32,
        };
        assert!(prepare(wide, Need::rows(32)).is_some());
        // The mix-collapse op takes no fused query, and the ordered norm's
        // entry is no tree entry.
        assert!(
            backend
                .sinkhorn_mix_collapse(&mix_norm_query(1e-5, 256), Need::rows(32))
                .is_err()
        );
        let ordered = mix_norm_tree::entry(
            PROFILE,
            Geometry {
                mix: SERVED.mix,
                capacity: 256,
            },
            sinkhorn_hc::mix_norm::norm_requirement(SERVED.mix, 1e-5),
        );
        assert!(
            Sm120::new()
                .with_sinkhorn_mix_norm(ordered)
                .sinkhorn_mix_collapse_norm(&mix_norm_query(1e-5, 256), Need::rows(32))
                .is_err()
        );
        // The served entry prepares.
        backend
            .sinkhorn_mix_collapse_norm(&mix_norm_query(1e-5, 256), Need::rows(256))
            .unwrap();
    }

    /// The tree entry's mean of squares as nvcc compiles it at `width`: one
    /// FMA by the exact reciprocal at a power of two, else the rounded
    /// quotient, then the rounded sum (see [`MIN_NORM_EPS_BITS`]).
    fn kernel_mean(s: f32, width: u32, eps: f32) -> f32 {
        if width.is_power_of_two() {
            s.mul_add(1.0 / width as f32, eps)
        } else {
            s / width as f32 + eps
        }
    }

    #[test]
    fn the_fused_norm_refuses_epsilons_its_kernel_rounds_otherwise() {
        // The review's counterexample: eps bits 1 and a collapsed row of
        // 2^-69 then zeros give s = 2^-138; the contract's mean is eps and
        // the kernel's 2 eps.
        let s = 2.0_f32.powi(-69) * 2.0_f32.powi(-69);
        let tiny = f32::from_bits(1);
        assert_eq!((s / 4096.0 + tiny).to_bits(), 1);
        assert_eq!(kernel_mean(s, 4096, tiny).to_bits(), 2);
        let least = f32::from_bits(MIN_NORM_EPS_BITS);
        assert_eq!(least, 2.0_f32.powi(-101));
        for (bits, served) in [
            (1, false),
            (MIN_NORM_EPS_BITS - 1, false),
            (MIN_NORM_EPS_BITS, true),
            (1e-5_f32.to_bits(), true),
        ] {
            let eps = f32::from_bits(bits);
            let entry = mix_norm_tree::entry(
                PROFILE,
                Geometry {
                    mix: SERVED.mix,
                    capacity: 32,
                },
                mix_norm_tree::norm_requirement(SERVED.mix, eps),
            );
            let prepared = Sm120::new()
                .with_sinkhorn_mix_norm(entry)
                .sinkhorn_mix_collapse_norm(&mix_norm_query(eps, 32), Need::rows(32));
            assert_eq!(prepared.is_ok(), served, "{bits:#x}");
            if !served {
                assert!(
                    matches!(prepared.err(), Some(Refused::Rule(rule)) if rule.contains("2^-101"))
                );
            }
        }
    }

    /// From the least served eps the kernel's mean equals the contract's
    /// at every width the tree entry takes: s sampled over every quotient
    /// that is not exact (s below 2^-126 H) and at that edge, at the least
    /// eps, its neighbours, wider ones and the served eps; below it the
    /// forms differ.
    #[test]
    fn from_the_least_eps_the_kernels_mean_is_the_contracts() {
        let least = MIN_NORM_EPS_BITS;
        let mut epsilons = vec![least, least + 1, least + 0x3f_ffff, least + 0x40_0000];
        epsilons.extend((0..64).map(|i| least + i * 0x0123_4567 % 0x0800_0000));
        epsilons.extend([1e-6_f32.to_bits(), 1e-5_f32.to_bits(), 1.0_f32.to_bits()]);
        for width in [1024_u32, 2048, 3072, 4096] {
            // s from zero to 2^-126 H, where the quotient turns normal.
            let top = (2.0_f32.powi(-126) * width as f32).to_bits();
            let samples = (0..=top).step_by(997).chain([top, top - 1]);
            for s in samples.map(f32::from_bits) {
                for &bits in &epsilons {
                    let eps = f32::from_bits(bits);
                    assert_eq!(
                        kernel_mean(s, width, eps).to_bits(),
                        (s / width as f32 + eps).to_bits(),
                        "width {width} s {:#x} eps {bits:#x}",
                        s.to_bits()
                    );
                }
            }
        }
        // Just below the least eps an inexact quotient that rounds up to
        // 2^-126 ties the sum: the contract rounds to even, the FMA down.
        let eps = f32::from_bits(least - 0x40_0000 + 1);
        let s = f32::from_bits((2.0_f32.powi(-126) * 4096.0).to_bits() - 1);
        assert_ne!(
            kernel_mean(s, 4096, eps).to_bits(),
            (s / 4096.0 + eps).to_bits()
        );
    }

    #[test]
    fn the_mix_norm_entry_is_chosen_whatever_the_registration_order() {
        let prepare = |order: &[(Profile, u32)]| {
            let backend = order
                .iter()
                .fold(Sm120::new(), |backend, &(profile, capacity)| {
                    backend.with_sinkhorn_mix_norm(mix_norm_entry(profile, capacity))
                });
            at_32(
                backend.sinkhorn_mix_collapse_norm(&mix_norm_query(1e-5, 32), Need::rows(32)),
                &mix_norm_buffers(),
            )
        };
        assert_eq!(
            symbols(&in_every_order(&registered(false), prepare)),
            ["chop_sinkhorn_hc_least_mix_collapse_norm_tree_v1"]
        );
        assert_eq!(
            in_every_order(&registered(true), prepare),
            Err(Refused::Rule(
                "two SM120 Sinkhorn mix-collapse and norm entries serve the op, chop_sinkhorn_hc_least_mix_collapse_norm_tree_v1 and chop_sinkhorn_hc_tied_mix_collapse_norm_tree_v1".into()
            ))
        );
    }

    /// The contract's tree norm (`sinkhorn_hyper_connection_oracle::
    /// input_norm`) is the tree entry's, whose CPU emulation the lab checks
    /// against the device: equal bits on finite rows, a partial last pass
    /// included.
    #[test]
    fn the_contract_tree_norm_is_the_tree_entrys() {
        use crate::row::rmsnorm_tree::{hf_norm, tree_sum};
        use kernel_core::codec::f32_to_bf16;
        use kernel_core::sinkhorn_hyper_connection_oracle::input_norm;
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |scale: f32| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32_to_bf16(((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale)
        };
        for width in [4096_u32, 7168, 512] {
            let mix = SinkhornMix {
                width,
                ..SERVED.mix
            };
            let query = Query::MixCollapseNorm {
                mix,
                rmsnorm_eps_bits: eps(),
                capacity: 8,
            };
            let w = width as usize;
            let weight = (0..w).map(|_| next(4.0)).collect::<Vec<_>>();
            // Ordinary rows, then one row's magnitudes spread wide.
            let mut rows = (0..3 * w).map(|_| next(2.0)).collect::<Vec<_>>();
            rows.extend((0..w).map(|i| next(if i % 97 == 0 { 3000.0 } else { 0.01 })));
            let expected = rows
                .chunks_exact(w)
                .flat_map(|row| hf_norm(row, &weight, 1e-5, tree_sum))
                .collect::<Vec<_>>();
            assert_eq!(
                input_norm(query, &rows, &weight).unwrap(),
                expected,
                "{width}"
            );
        }
    }

    #[test]
    fn launch_keys_hold_what_tiled_and_fused_entries_launch() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let base = tiled::entry(PROFILE, SERVED, 4);
        let names = field_names!(tiled::Entry {
            profile,
            symbol,
            query,
            rows,
            probe,
            params
        });
        let fields: Vec<Field<tiled::Entry>> = vec![
            ("profile", Class::Inert, Box::new(|e| e.profile = WIDE)),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_sinkhorn_hc_other_tiled_v1".into()),
            ),
            (
                "query",
                Class::Inert,
                Box::new(|e| e.query = mix_collapse(512)),
            ),
            ("rows", Class::Launch, Box::new(|e| e.rows = 8)),
            ("probe", Class::Launch, Box::new(|e| e.probe = true)),
            ("params", Class::Launch, Box::new(|e| e.params.swap(0, 1))),
        ];
        // Beside the split entry, at the split's rows and the tile's.
        keys_follow_launches(&base, names, fields, tiled_launch, |e| {
            let prepared = backend(true, false)
                .with_sinkhorn_tiled([e.clone()])
                .sinkhorn_mix_collapse(&mix_collapse(256), Need::rows(256))?;
            let launches = [1, 143, 144, 256].map(|rows| {
                prepared
                    .launches(&mix_buffers(), View::EMPTY, Counts::rows(rows))
                    .unwrap()
            });
            Ok((format!("{:?} {:?}", e.body(), e.params), launches))
        });
        let base = mix_norm_entry(PROFILE, 256);
        let names = field_names!(mix_norm_tree::Entry {
            profile,
            symbol,
            query,
            norm,
            params
        });
        let fields: Vec<Field<mix_norm_tree::Entry>> = vec![
            ("profile", Class::Inert, Box::new(|e| e.profile = WIDE)),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_sinkhorn_hc_other_norm_v1".into()),
            ),
            (
                "query",
                Class::Inert,
                Box::new(|e| e.query = mix_collapse(512)),
            ),
            (
                "norm",
                Class::Selects,
                Box::new(|e| e.norm.epsilon_bits += 1),
            ),
            ("params", Class::Launch, Box::new(|e| e.params.swap(0, 1))),
        ];
        keys_follow_launches(&base, names, fields, mix_norm_launch, |e| {
            let prepared = Sm120::new()
                .with_sinkhorn_mix_norm(e.clone())
                .sinkhorn_mix_collapse_norm(&mix_norm_query(1e-5, 256), Need::rows(32))?;
            let launches = [1, 32].map(|rows| {
                prepared
                    .launches(&mix_norm_buffers(), View::EMPTY, Counts::rows(rows))
                    .unwrap()
            });
            Ok((format!("{:?} {:?}", e.body(), e.params), launches))
        });
    }

    #[test]
    fn expand_and_head_launch_their_entries_as_their_builders_do() {
        let entries = sinkhorn_hc::entries(PROFILE, SERVED, false);
        let backend = backend(true, true);
        let expand = backend
            .sinkhorn_expand(
                &Query::Expand {
                    streams: 4,
                    width: 4096,
                    capacity: 8192,
                },
                Need::rows(8192),
            )
            .unwrap();
        let head = backend
            .sinkhorn_head(
                &Query::Head {
                    streams: 4,
                    width: 4096,
                    mode: StreamHeadMode::MeanOfStreams,
                    capacity: 8192,
                },
                Need::rows(8192),
            )
            .unwrap();
        let expand_buffers = StreamExpandBuffers {
            x: view(1, u64::MAX),
            y: view(2, u64::MAX),
            post: view(3, u64::MAX),
            comb: view(4, u64::MAX),
            output: view(5, u64::MAX),
            live_rows: view(6, 4),
        };
        let head_buffers = StreamHeadBuffers {
            x: view(1, u64::MAX),
            output: view(2, u64::MAX),
            live_rows: view(3, 4),
        };
        for rows in [1, 32, 6 * 32, 8192] {
            let launch = |entry: &sinkhorn_hc::Entry, last| Launch {
                symbol: entry.symbol.clone(),
                rows,
                plan: entry.plan(rows, rows).unwrap(),
                arguments: entry.pack_arguments(handles(1..=last)).unwrap(),
                collective: false,
                persistent: false,
            };
            assert_eq!(
                expand
                    .launches(&expand_buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                [launch(&entries[1], 6)]
            );
            assert_eq!(
                head.launches(&head_buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap(),
                [launch(&entries[2], 3)]
            );
        }
        assert_eq!(expand.info().implementation, "sm120.sinkhorn.expand.v1");
        assert_eq!(head.info().implementation, "sm120.sinkhorn.head.v1");
        assert!(expand.info().weight_layouts.is_empty());
    }

    fn reduce_query(ranks: u32, capacity: u32) -> Query {
        Query::AllReduceExpandMixNorm {
            ranks,
            dtype: DType::F32,
            mix: SERVED.mix,
            rmsnorm_eps_bits: eps(),
            norm: RmsNormReduction::Ordered,
            capacity,
        }
    }

    /// Rank `r`'s output is handle 10 + r and its state regions
    /// 100 * (r + 1) + i; the epilogue's views are handles 20 to 33.
    fn reduce_buffers(ranks: u32, state: &[u64]) -> ReduceExpandMixNormBuffers<View> {
        ReduceExpandMixNormBuffers {
            reduce: AllReduceBuffers {
                input: view(1, u64::MAX),
                output: view(2, u64::MAX),
                live_rows: None,
                ranks: (0..ranks)
                    .map(|r| RankViews {
                        output: view(10 + u64::from(r), u64::MAX),
                        state: state
                            .iter()
                            .enumerate()
                            .map(|(i, &bytes)| view(100 * (u64::from(r) + 1) + i as u64, bytes))
                            .collect(),
                    })
                    .collect(),
            },
            streams: view(20, u64::MAX),
            reduced: view(21, u64::MAX),
            expanded: view(22, u64::MAX),
            mixer: mixer(23),
            post: view(26, u64::MAX),
            comb: view(27, u64::MAX),
            collapsed: view(28, u64::MAX),
            norm_weight: view(29, 8192),
            normed: view(30, u64::MAX),
            scratch: scratch(31),
            live_rows: view(33, 4),
        }
    }

    #[test]
    fn the_comm_epilogue_follows_the_all_reduce_as_its_builders_launch_them() {
        let (ranks, rank, capacity) = (4, 1, 512);
        let entry = sinkhorn_split::entry(PROFILE, SERVED.mix, eps(), SERVED.capacity).unwrap();
        let prepared = backend(true, true)
            .sinkhorn_reduce_expand_mix_norm(
                &reduce_query(ranks, capacity),
                rank,
                Need::rows(capacity),
            )
            .unwrap();
        let sum = CommQuery::AllReduce {
            ranks,
            elements: 4096,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        let reduce = Sm120::new()
            .all_reduce(&sum, rank, Need::rows(capacity))
            .unwrap();
        let info = prepared.info();
        assert_eq!(info.state, reduce.info().state);
        let mut symbols = reduce.info().symbols.clone();
        symbols.push((entry.symbol.clone(), 0));
        assert_eq!(info.symbols, symbols);
        assert_eq!(
            info.implementation,
            format!("{}+{EXPAND_MIX_NORM}", reduce.info().implementation)
        );
        assert_eq!(
            info.bytes_of("scratch.partials"),
            Some(split::partials_bytes(SERVED.mix, capacity))
        );
        let buffers = reduce_buffers(ranks, &info.state);
        let epilogue = EpilogueBuffers {
            streams: 20,
            post: 26,
            comb: 27,
            reduced: 21,
            expanded: 22,
            fn_: 23,
            base: 24,
            scale: 25,
            collapsed: 28,
            weight: 29,
            normed: 30,
            live_rows: 33,
            eps_bits: eps(),
        };
        let split = SplitBuffers {
            epilogue,
            partials: 31,
            tickets: 32,
        };
        let mut variants = Vec::new();
        for rows in [1, 2, 32, 6 * 32, capacity] {
            let mut launches = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            let last = launches.pop().unwrap();
            // The all-reduce's own launches, collective, into the sums the
            // epilogue reads.
            assert_eq!(
                launches,
                reduce
                    .launches(&buffers.reduce, View::EMPTY, Counts::rows(rows))
                    .unwrap()
            );
            assert!(launches.iter().all(|launch| launch.collective));
            variants.push(launches.len());
            assert_eq!(
                last,
                Launch {
                    symbol: entry.symbol.clone(),
                    rows,
                    plan: entry.plan(rows, ranks).unwrap(),
                    arguments: sinkhorn_split::rows_arguments(2, split).unwrap(),
                    collective: false,
                    persistent: false,
                }
            );
        }
        // Both the one-shot and the phased all-reduce.
        assert_eq!(variants.first(), Some(&1));
        assert_eq!(variants.last(), Some(&2));
    }

    #[test]
    fn prepare_refuses_unserved_over_limit_and_inexact_ops() {
        let backend = backend(true, true);
        let rule = |result: Result<Prepared<MixCollapseBuffers<View>>, Refused>| match result {
            Err(Refused::Rule(rule)) => rule,
            other => panic!("expected a rule refusal, got {:?}", other.err()),
        };
        // Another op's query, a mixer no entry serves, and the declared
        // but unsupported variants.
        let expand = Query::Expand {
            streams: 4,
            width: 4096,
            capacity: 8,
        };
        rule(backend.sinkhorn_mix_collapse(&expand, Need::rows(8)));
        let other = Query::MixCollapse {
            mix: SinkhornMix {
                iterations: 10,
                ..SERVED.mix
            },
            capacity: 8,
        };
        assert!(rule(backend.sinkhorn_mix_collapse(&other, Need::rows(8))).contains("no SM120"));
        assert!(matches!(
            backend.sinkhorn_head(
                &Query::Head {
                    streams: 4,
                    width: 4096,
                    mode: StreamHeadMode::SigmoidWeighted,
                    capacity: 8,
                },
                Need::rows(8)
            ),
            Err(Refused::Rule(_))
        ));
        assert!(matches!(
            backend.sinkhorn_reduce_expand_mix_norm(&mix_collapse(8), 0, Need::rows(8)),
            Err(Refused::Rule(_))
        ));
        // Without the entries, nothing is served.
        assert!(matches!(
            Sm120::new().sinkhorn_reduce_expand_mix_norm(&reduce_query(4, 8), 0, Need::rows(8)),
            Err(Refused::Rule(_))
        ));
        // SM120's limits: eight streams of 4096 exceed the one-CTA entry's
        // shared-memory row.
        let wide = SinkhornMix {
            streams: 8,
            ..SERVED.mix
        };
        let geometry = Geometry {
            mix: wide,
            capacity: 32,
        };
        let over = Sm120::new().with_sinkhorn(sinkhorn_hc::entries(PROFILE, geometry, false));
        let query = Query::MixCollapse {
            mix: wide,
            capacity: 32,
        };
        assert!(rule(over.sinkhorn_mix_collapse(&query, Need::rows(8))).contains("40 KiB"));
        // Capacities past the query's or the entry's rows.
        assert!(
            rule(backend.sinkhorn_mix_collapse(&mix_collapse(8), Need::rows(9))).contains("8 rows")
        );
        assert!(
            rule(backend.sinkhorn_mix_collapse(&mix_collapse(8), Need::rows(0))).contains("8 rows")
        );
        assert!(
            rule(backend.sinkhorn_mix_collapse(&mix_collapse(8193), Need::rows(8193)))
                .contains("at most 8192")
        );
        // The query's bound alone refuses nothing: its prepared rows choose.
        assert!(
            backend
                .sinkhorn_mix_collapse(&mix_collapse(8193), Need::rows(8))
                .is_ok()
        );
        // Exactness no Sinkhorn entry declares (a tie rule), and the comm
        // epilogue after the kernel base's F32 all-reduce, which declares
        // nothing; the declared one-row bits are granted.
        let ties = Need::rows(8).with_exact(Invariance::one_row(8).with_ties(Ties::LowerIndex));
        assert!(matches!(
            backend.sinkhorn_mix_collapse(&mix_collapse(8), ties),
            Err(Refused::Inexact { .. })
        ));
        assert!(matches!(
            backend.sinkhorn_expand(&expand, ties),
            Err(Refused::Inexact { .. })
        ));
        let exact = Need::rows(8).with_exact(Invariance::one_row(8));
        assert!(
            backend
                .sinkhorn_mix_collapse(&mix_collapse(8), exact)
                .is_ok()
        );
        assert!(backend.sinkhorn_expand(&expand, exact).is_ok());
        assert!(matches!(
            backend.sinkhorn_reduce_expand_mix_norm(&reduce_query(4, 8), 0, exact),
            Err(Refused::Inexact { .. })
        ));
    }

    /// After the BF16-partials all-reduce, the comm epilogue claims what
    /// the all-reduce's forms, the widen row and the split entry declare
    /// together, and refuses what its rounding rows do not declare.
    #[test]
    fn the_bf16_comm_epilogue_claims_its_launches_together() {
        use crate::row::rounding;
        let width = u64::from(SERVED.mix.width);
        let query = Query::AllReduceExpandMixNorm {
            ranks: 4,
            dtype: DType::Bf16,
            mix: SERVED.mix,
            rmsnorm_eps_bits: eps(),
            norm: RmsNormReduction::Ordered,
            capacity: 8192,
        };
        let over = |pair: rounding::PartialRounding, phased: bool| {
            let backend = backend(false, true).with_partial_rounding(pair);
            match phased {
                true => backend.with_phased_bf16_all_reduce(),
                false => backend,
            }
        };
        let shared = rounding::shared(width as u32);
        let exact = Need::rows(8192).with_exact(Invariance::any_rows(32));
        for phased in [true, false] {
            let op = over(shared.clone(), phased)
                .sinkhorn_reduce_expand_mix_norm(&query, 0, exact)
                .unwrap();
            assert_eq!(op.info().guarantees, Invariance::any_rows(8192), "{phased}");
            let undeclared = rounding::PartialRounding {
                invariance: Invariance::NONE,
                ..shared.clone()
            };
            assert!(
                matches!(
                    over(undeclared, phased).sinkhorn_reduce_expand_mix_norm(&query, 0, exact),
                    Err(Refused::Inexact { .. })
                ),
                "{phased}"
            );
        }
    }

    /// An epilogue entry serves the norm order its kernel sums in and no
    /// other: the column-order split entry refuses a tree query, the tree
    /// entry an ordered one, and each serves its own.
    #[test]
    fn comm_epilogues_serve_only_their_norm_order() {
        let ordered = sinkhorn_split::catalog();
        let tree = crate::comm::sinkhorn_split_tree::catalog();
        let query = |norm| Query::AllReduceExpandMixNorm {
            ranks: 4,
            dtype: DType::F32,
            mix: SERVED.mix,
            rmsnorm_eps_bits: eps(),
            norm,
            capacity: 32,
        };
        let tree_order = RmsNormReduction::WarpTree { threads: 256 };
        for (entry, served, other) in [
            (ordered, RmsNormReduction::Ordered, tree_order),
            (tree, tree_order, RmsNormReduction::Ordered),
        ] {
            let backend = backend(false, false).with_sinkhorn_epilogue(entry.clone());
            let op = backend
                .sinkhorn_reduce_expand_mix_norm(&query(served), 0, Need::rows(32))
                .unwrap();
            assert!(
                op.info()
                    .symbols
                    .iter()
                    .any(|(symbol, _)| *symbol == entry.symbol),
                "{served:?}"
            );
            assert!(
                backend
                    .sinkhorn_reduce_expand_mix_norm(&query(other), 0, Need::rows(32))
                    .is_err(),
                "{} serves {other:?}",
                entry.symbol
            );
        }
    }

    #[test]
    fn prepare_refuses_valid_math_past_the_sm120_kernels() {
        let backend = backend(true, true);
        let refused = |result: Result<(), Refused>| match result {
            Err(Refused::Rule(rule)) => rule,
            other => panic!("expected a rule refusal, got {other:?}"),
        };
        let mixer = |streams, width, iterations| SinkhornMix {
            streams,
            width,
            iterations,
            ..SERVED.mix
        };
        // Widths off the 128-K chains or past 16384, and over 64 iterations.
        for mix in [mixer(4, 4128, 20), mixer(2, 16_512, 20), mixer(4, 4096, 65)] {
            let query = Query::MixCollapse { mix, capacity: 8 };
            query.validate().unwrap();
            let rule = refused(
                backend
                    .sinkhorn_mix_collapse(&query, Need::rows(8))
                    .map(drop),
            );
            assert!(rule.contains("128-K chains"), "{rule}");
            let fused = Query::AllReduceExpandMixNorm {
                ranks: 4,
                dtype: DType::F32,
                mix,
                rmsnorm_eps_bits: eps(),
                norm: RmsNormReduction::Ordered,
                capacity: 8,
            };
            fused.validate().unwrap();
            assert!(
                refused(
                    backend
                        .sinkhorn_reduce_expand_mix_norm(&fused, 0, Need::rows(8))
                        .map(drop)
                )
                .contains("128-K chains")
            );
        }
        // One stream or more than eight.
        for streams in [1, 9] {
            let query = Query::MixCollapse {
                mix: mixer(streams, 1024, 20),
                capacity: 8,
            };
            query.validate().unwrap();
            assert_eq!(
                refused(
                    backend
                        .sinkhorn_mix_collapse(&query, Need::rows(8))
                        .map(drop)
                ),
                "SM120 Sinkhorn kernels take 2 to 8 streams"
            );
            let expand = Query::Expand {
                streams,
                width: 4096,
                capacity: 8,
            };
            let head = Query::Head {
                streams,
                width: 4096,
                mode: StreamHeadMode::MeanOfStreams,
                capacity: 8,
            };
            assert!(
                refused(backend.sinkhorn_expand(&expand, Need::rows(8)).map(drop))
                    .contains("streams")
            );
            assert!(
                refused(backend.sinkhorn_head(&head, Need::rows(8)).map(drop)).contains("streams")
            );
        }
        // More ranks than the comm pointer arrays hold.
        let wide = reduce_query(9, 8);
        wide.validate().unwrap();
        assert_eq!(
            refused(
                backend
                    .sinkhorn_reduce_expand_mix_norm(&wide, 0, Need::rows(8))
                    .map(drop)
            ),
            "the Sinkhorn comm epilogue's pointer arrays hold 8 ranks"
        );
    }

    #[test]
    fn expand_and_head_prepare_stream_rows_past_u32_elements() {
        // Eight streams of 2^29 columns, sized in u64.
        let mix = SinkhornMix {
            streams: 8,
            width: 1 << 29,
            ..SERVED.mix
        };
        let entries = sinkhorn_hc::entries(PROFILE, Geometry { mix, capacity: 1 }, false);
        let backend = Sm120::new().with_sinkhorn(entries.clone());
        let expand = backend
            .sinkhorn_expand(
                &Query::Expand {
                    streams: 8,
                    width: 1 << 29,
                    capacity: 1,
                },
                Need::rows(1),
            )
            .unwrap();
        let wide = 8 << 30;
        let buffers = StreamExpandBuffers {
            x: view(1, wide),
            y: view(2, 1 << 30),
            post: view(3, 16),
            comb: view(4, 128),
            output: view(5, wide),
            live_rows: view(6, 4),
        };
        let launches = expand
            .launches(&buffers, View::EMPTY, Counts::rows(1))
            .unwrap();
        assert_eq!(launches[0].plan, entries[1].plan(1, 1).unwrap());
        assert_eq!(launches[0].plan.grid, [1 << 20, 1, 1]);
        let short = StreamExpandBuffers {
            output: view(5, wide - 1),
            ..buffers
        };
        assert!(
            expand
                .launches(&short, View::EMPTY, Counts::rows(1))
                .is_err()
        );
        let head = backend
            .sinkhorn_head(
                &Query::Head {
                    streams: 8,
                    width: 1 << 29,
                    mode: StreamHeadMode::MeanOfStreams,
                    capacity: 1,
                },
                Need::rows(1),
            )
            .unwrap();
        let buffers = StreamHeadBuffers {
            x: view(1, wide),
            output: view(2, 1 << 30),
            live_rows: view(3, 4),
        };
        assert_eq!(
            head.launches(&buffers, View::EMPTY, Counts::rows(1))
                .unwrap()[0]
                .plan,
            entries[2].plan(1, 1).unwrap()
        );
    }

    /// Profiles whose entries hold 8, 32 (twice: `LEAST` and the
    /// different `TIED`) and 64 rows, for an op of 32 rows.
    const NARROW: Profile = Profile::synthetic(ProfileNames {
        name: "sinkhorn-narrow",
        tag: "narrow",
    });
    const LEAST: Profile = Profile::synthetic(ProfileNames {
        name: "sinkhorn-least",
        tag: "least",
    });
    const TIED: Profile = Profile::synthetic(ProfileNames {
        name: "sinkhorn-tied",
        tag: "tied",
    });
    const WIDE: Profile = Profile::synthetic(ProfileNames {
        name: "sinkhorn-wide",
        tag: "wide",
    });

    /// The registered sets: an entry too narrow for the rows, the least
    /// that holds them, an identical copy of it, and a wider one; with
    /// `tied`, also a different entry of the least capacity.
    fn registered(tied: bool) -> Vec<(Profile, u32)> {
        let mut set = vec![(NARROW, 8), (LEAST, 32), (LEAST, 32), (WIDE, 64)];
        if tied {
            set.push((TIED, 32));
        }
        set
    }

    /// Every order of `items`.
    fn orders<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
        if items.is_empty() {
            return vec![Vec::new()];
        }
        let mut all = Vec::new();
        for i in 0..items.len() {
            let mut rest = items.to_vec();
            let first = rest.remove(i);
            for mut order in orders(&rest) {
                order.insert(0, first.clone());
                all.push(order);
            }
        }
        all
    }

    type Prepared32 = Result<(OpInfo, Vec<Launch>), Refused>;
    type Prepare<'a> = &'a dyn Fn(&[(Profile, u32)]) -> Prepared32;

    /// The op's info and launches at 32 rows, or its refusal.
    fn at_32<B>(prepared: Result<Prepared<B>, Refused>, buffers: &B) -> Prepared32 {
        let prepared = prepared?;
        let launches = prepared
            .launches(buffers, View::EMPTY, Counts::rows(32))
            .unwrap();
        Ok((prepared.info().clone(), launches))
    }

    /// What `prepare` gives for every registration order of `set`, which
    /// must be the same.
    fn in_every_order(
        set: &[(Profile, u32)],
        prepare: impl Fn(&[(Profile, u32)]) -> Prepared32,
    ) -> Prepared32 {
        let all = orders(set);
        let first = prepare(&all[0]);
        for order in &all[1..] {
            assert_eq!(prepare(order), first, "{order:?}");
        }
        first
    }

    fn symbols(prepared: &Prepared32) -> Vec<String> {
        let (info, launches) = prepared.as_ref().unwrap();
        for launch in launches {
            assert!(info.symbols.iter().any(|(s, _)| *s == launch.symbol));
        }
        launches.iter().map(|l| l.symbol.clone()).collect()
    }

    #[test]
    fn the_split_mix_collapse_is_chosen_whatever_the_registration_order() {
        let prepare = |order: &[(Profile, u32)]| {
            // The served one-CTA entries too: the split entry still serves.
            let mut backend =
                Sm120::new().with_sinkhorn(sinkhorn_hc::entries(PROFILE, SERVED, false));
            for &(profile, capacity) in order {
                let geometry = Geometry {
                    mix: SERVED.mix,
                    capacity,
                };
                backend = backend.with_sinkhorn_split(split::entry(profile, geometry));
            }
            at_32(
                backend.sinkhorn_mix_collapse(&mix_collapse(32), Need::rows(32)),
                &mix_buffers(),
            )
        };
        assert_eq!(
            symbols(&in_every_order(&registered(false), prepare)),
            ["chop_sinkhorn_hc_least_mix_collapse_split_v1"]
        );
        assert_eq!(
            in_every_order(&registered(true), prepare),
            Err(Refused::Rule(
                "two SM120 Sinkhorn split mix-collapse entries serve the op, chop_sinkhorn_hc_least_mix_collapse_split_v1 and chop_sinkhorn_hc_tied_mix_collapse_split_v1".into()
            ))
        );
        // A split entry too narrow for the rows leaves the one-CTA entry.
        assert_eq!(
            symbols(&prepare(&[(NARROW, 8)])),
            ["chop_sinkhorn_hc_test_mix_collapse_v1"]
        );
    }

    #[test]
    fn one_cta_entries_are_chosen_whatever_the_registration_order() {
        let backend = |order: &[(Profile, u32)]| {
            order
                .iter()
                .fold(Sm120::new(), |backend, &(profile, capacity)| {
                    let geometry = Geometry {
                        mix: SERVED.mix,
                        capacity,
                    };
                    backend.with_sinkhorn(sinkhorn_hc::entries(profile, geometry, false))
                })
        };
        let mix = |order: &[(Profile, u32)]| {
            let buffers = MixCollapseBuffers {
                scratch: MixScratch {
                    partials: None,
                    tickets: None,
                },
                ..mix_buffers()
            };
            at_32(
                backend(order).sinkhorn_mix_collapse(&mix_collapse(32), Need::rows(32)),
                &buffers,
            )
        };
        let expand = |order: &[(Profile, u32)]| {
            let query = Query::Expand {
                streams: 4,
                width: 4096,
                capacity: 32,
            };
            let buffers = StreamExpandBuffers {
                x: view(1, u64::MAX),
                y: view(2, u64::MAX),
                post: view(3, u64::MAX),
                comb: view(4, u64::MAX),
                output: view(5, u64::MAX),
                live_rows: view(6, 4),
            };
            at_32(
                backend(order).sinkhorn_expand(&query, Need::rows(32)),
                &buffers,
            )
        };
        let head = |order: &[(Profile, u32)]| {
            let query = Query::Head {
                streams: 4,
                width: 4096,
                mode: StreamHeadMode::MeanOfStreams,
                capacity: 32,
            };
            let buffers = StreamHeadBuffers {
                x: view(1, u64::MAX),
                output: view(2, u64::MAX),
                live_rows: view(3, 4),
            };
            at_32(
                backend(order).sinkhorn_head(&query, Need::rows(32)),
                &buffers,
            )
        };
        let ops: [(&str, &str, Prepare); 3] = [
            ("mix_collapse", "mix-collapse", &mix),
            ("expand", "Expand", &expand),
            ("head", "Head", &head),
        ];
        for (entry, op, prepare) in ops {
            assert_eq!(
                symbols(&in_every_order(&registered(false), prepare)),
                [format!("chop_sinkhorn_hc_least_{entry}_v1")]
            );
            assert_eq!(
                in_every_order(&registered(true), prepare),
                Err(Refused::Rule(format!(
                    "two SM120 Sinkhorn {op} entries serve the op, chop_sinkhorn_hc_least_{entry}_v1 and chop_sinkhorn_hc_tied_{entry}_v1"
                )))
            );
        }
    }

    /// An entry registered again under another profile launches the same,
    /// so it is one entry in every order.
    #[test]
    fn profile_only_registrations_are_one_entry() {
        let entry = sinkhorn_hc::entries(PROFILE, SERVED, false)
            .into_iter()
            .find(|entry| entry.op == Op::Expand)
            .unwrap();
        let other = sinkhorn_hc::Entry {
            profile: Profile::synthetic(ProfileNames {
                name: "sinkhorn-backend-other",
                tag: "sinkhorn_backend_other",
            }),
            ..entry.clone()
        };
        let Query::Expand { streams, width, .. } = entry.query else {
            unreachable!()
        };
        let query = Query::Expand {
            streams,
            width,
            capacity: capacity(entry.query),
        };
        let prepare = |order: Vec<sinkhorn_hc::Entry>| {
            Sm120::new()
                .with_sinkhorn(order)
                .sinkhorn_expand(&query, Need::rows(1))
                .map(|op| op.info().clone())
        };
        let alone = prepare(vec![entry.clone()]).unwrap();
        assert_eq!(
            prepare(vec![entry.clone(), other.clone()]),
            Ok(alone.clone())
        );
        assert_eq!(prepare(vec![other, entry]), Ok(alone));
    }

    /// The expand entries of `order`'s profiles and capacities.
    fn expand_entries(order: &[(Profile, u32)]) -> Vec<sinkhorn_hc::Entry> {
        order
            .iter()
            .flat_map(|&(profile, capacity)| {
                let geometry = Geometry {
                    mix: SERVED.mix,
                    capacity,
                };
                sinkhorn_hc::entries(profile, geometry, false)
            })
            .filter(|entry| entry.op == Op::Expand)
            .collect()
    }

    /// An expand of 4 streams of 4096 bounded at `capacity` rows.
    fn expand_query(capacity: u32) -> Query {
        Query::Expand {
            streams: 4,
            width: 4096,
            capacity,
        }
    }

    /// The prepared rows choose the entry, not the query's row bound: a
    /// plan depends on its bucket's rows alone.
    #[test]
    fn the_least_capacity_holding_the_prepared_rows_serves() {
        let buffers = StreamExpandBuffers {
            x: view(1, u64::MAX),
            y: view(2, u64::MAX),
            post: view(3, u64::MAX),
            comb: view(4, u64::MAX),
            output: view(5, u64::MAX),
            live_rows: view(6, 4),
        };
        let prepare = |order: &[(Profile, u32)], rows: u32| {
            let prepared = Sm120::new()
                .with_sinkhorn(expand_entries(order))
                .sinkhorn_expand(&expand_query(32), Need::rows(rows))?;
            let launches = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            Ok::<_, Refused>((prepared.info().clone(), launches))
        };
        let set = [(NARROW, 8), (LEAST, 32)];
        for order in orders(&set) {
            let (info, launches) = prepare(&order, 8).unwrap();
            assert_eq!(info.symbols[0].0, "chop_sinkhorn_hc_narrow_expand_v1");
            assert_eq!(launches[0].symbol, info.symbols[0].0);
            assert_eq!(
                prepare(&order, 32).unwrap().0.symbols[0].0,
                "chop_sinkhorn_hc_least_expand_v1"
            );
        }
        // The entry of 8 rows serves 8 rows of a query bounded at 32 alone.
        assert_eq!(
            prepare(&[(NARROW, 8)], 8).unwrap().0.symbols[0].0,
            "chop_sinkhorn_hc_narrow_expand_v1"
        );
        assert!(prepare(&[(NARROW, 8)], 32).is_err());
    }

    /// A need with claims is refused as inexact in every order, before any
    /// tie between entries that would not keep it either.
    #[test]
    fn exact_needs_are_refused_before_ties() {
        let need = Need::rows(8).with_exact(Invariance::one_row(8).with_ties(Ties::LowerIndex));
        for set in [vec![(LEAST, 32)], vec![(LEAST, 32), (TIED, 32)]] {
            for order in orders(&set) {
                let prepared = Sm120::new()
                    .with_sinkhorn(expand_entries(&order))
                    .sinkhorn_expand(&expand_query(32), need)
                    .map(|op| op.info().clone());
                assert_eq!(
                    prepared,
                    Err(Refused::Inexact {
                        needed: need.exact,
                        guaranteed: Invariance::any_rows(32),
                    }),
                    "{order:?}"
                );
            }
        }
    }

    #[test]
    fn the_comm_epilogue_is_chosen_whatever_the_registration_order() {
        let query = reduce_query(4, 32);
        let state = backend(true, true)
            .sinkhorn_reduce_expand_mix_norm(&query, 1, Need::rows(32))
            .unwrap()
            .info()
            .state
            .clone();
        let buffers = reduce_buffers(4, &state);
        let prepare = |order: &[(Profile, u32)]| {
            let backend = order
                .iter()
                .fold(Sm120::new(), |backend, &(profile, capacity)| {
                    backend.with_sinkhorn_epilogue(
                        sinkhorn_split::entry(profile, SERVED.mix, eps(), capacity).unwrap(),
                    )
                });
            at_32(
                backend.sinkhorn_reduce_expand_mix_norm(&query, 1, Need::rows(32)),
                &buffers,
            )
        };
        let symbols = symbols(&in_every_order(&registered(false), prepare));
        assert_eq!(
            symbols.last().map(String::as_str),
            Some("chop_sinkhorn_epilogue_least_rows_split_v1")
        );
        assert_eq!(
            in_every_order(&registered(true), prepare),
            Err(Refused::Rule(
                "two SM120 Sinkhorn comm epilogue entries serve the op, chop_sinkhorn_epilogue_least_rows_split_v1 and chop_sinkhorn_epilogue_tied_rows_split_v1".into()
            ))
        );
    }

    /// The comm epilogue's all-reduce takes the model's low-latency
    /// registration as the plain all-reduce does: the same state and the
    /// same reduce launches at every row count.
    #[test]
    fn the_comm_epilogue_reduces_as_the_plain_all_reduce() {
        let backend = backend(true, true).with_low_latency_all_reduce(8);
        let prepared = backend
            .sinkhorn_reduce_expand_mix_norm(&reduce_query(4, 8), 1, Need::rows(8))
            .unwrap();
        let sum = CommQuery::AllReduce {
            ranks: 4,
            elements: u64::from(SERVED.mix.width),
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        let plain = backend.all_reduce(&sum, 1, Need::rows(8)).unwrap();
        assert_eq!(prepared.info().state, plain.info().state);
        let buffers = reduce_buffers(4, &prepared.info().state);
        for rows in 1..=8 {
            let launches = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(rows))
                .unwrap();
            let reduced = plain
                .launches(&buffers.reduce, View::EMPTY, Counts::rows(rows))
                .unwrap();
            assert_eq!(launches[..reduced.len()], reduced[..], "{rows} rows");
            let low_latency = launches[0].symbol == crate::comm::ll_f32::SYMBOL;
            assert_eq!(low_latency, rows <= 2, "{rows} rows");
        }
    }

    /// Both norm orders' epilogues serve epsilons from 2^-101 on, the bound
    /// where their mean's FMA equals the contract's two roundings.
    #[test]
    fn the_comm_epilogue_refuses_epsilons_its_norm_rounds_otherwise() {
        let tree_order = RmsNormReduction::WarpTree { threads: 256 };
        for (entry, norm) in [
            (sinkhorn_split::catalog(), RmsNormReduction::Ordered),
            (crate::comm::sinkhorn_split_tree::catalog(), tree_order),
        ] {
            for (bits, served) in [
                (1, false),
                (MIN_NORM_EPS_BITS - 1, false),
                (MIN_NORM_EPS_BITS, true),
                (eps(), true),
            ] {
                let entry = sinkhorn_split::Entry {
                    rmsnorm_eps_bits: bits,
                    ..entry.clone()
                };
                let query = Query::AllReduceExpandMixNorm {
                    ranks: 4,
                    dtype: DType::F32,
                    mix: SERVED.mix,
                    rmsnorm_eps_bits: bits,
                    norm,
                    capacity: 32,
                };
                let prepared = backend(false, false)
                    .with_sinkhorn_epilogue(entry)
                    .sinkhorn_reduce_expand_mix_norm(&query, 0, Need::rows(32));
                assert_eq!(prepared.is_ok(), served, "{norm:?} {bits:#x}");
                if !served {
                    assert!(
                        matches!(prepared.err(), Some(Refused::Rule(rule)) if rule.contains("2^-101"))
                    );
                }
            }
        }
    }

    #[test]
    fn the_comm_epilogue_refuses_what_its_builder_refuses() {
        // A rank outside the ranks, as the all-reduce refuses.
        let backend = backend(true, true);
        assert!(matches!(
            backend.sinkhorn_reduce_expand_mix_norm(&reduce_query(4, 8), 4, Need::rows(8)),
            Err(Refused::Rule(_))
        ));
        // A mixer wider than the fused width, which the contract allows.
        let wide = SinkhornMix {
            width: 5120,
            ..SERVED.mix
        };
        let entry = sinkhorn_split::Entry {
            symbol: "chop_sinkhorn_epilogue_test_rows_split_v1".into(),
            mix: wide,
            rmsnorm_eps_bits: eps(),
            norm: RmsNormReduction::Ordered,
            capacity: 32,
            params: sinkhorn_split::params(DType::Bf16),
        };
        let query = Query::AllReduceExpandMixNorm {
            ranks: 4,
            dtype: DType::F32,
            mix: wide,
            rmsnorm_eps_bits: eps(),
            norm: RmsNormReduction::Ordered,
            capacity: 32,
        };
        query.validate().unwrap();
        assert_eq!(
            Sm120::new()
                .with_sinkhorn_epilogue(entry)
                .sinkhorn_reduce_expand_mix_norm(&query, 0, Need::rows(8))
                .err(),
            Some(Refused::Rule(
                "the Sinkhorn comm epilogue runs at the fused width only".into()
            ))
        );
    }

    #[test]
    fn launches_check_every_view_and_count() {
        let backend = backend(true, true);
        let prepared = backend
            .sinkhorn_mix_collapse(&mix_collapse(32), Need::rows(32))
            .unwrap();
        let buffers = mix_buffers();
        let sized = |rows: u32| MixCollapseBuffers {
            x: view(1, u64::from(rows) * 16_384 * 2),
            mixer: MixerWeights {
                fn_: view(2, 24 * 16_384 * 2),
                base: view(3, 24 * 4),
                scale: view(4, 12),
            },
            collapsed: view(5, u64::from(rows) * 4096 * 2),
            post: view(6, u64::from(rows) * 4 * 2),
            comb: view(7, u64::from(rows) * 16 * 2),
            scratch: MixScratch {
                partials: Some(view(8, split::partials_bytes(SERVED.mix, rows))),
                tickets: Some(view(9, u64::from(rows) * 4)),
            },
            live_rows: view(10, 4),
        };
        prepared
            .launches(&sized(32), View::EMPTY, Counts::rows(32))
            .unwrap();
        let shrink = |edit: fn(&mut MixCollapseBuffers<View>)| {
            let mut short = sized(32);
            edit(&mut short);
            prepared
                .launches(&short, View::EMPTY, Counts::rows(32))
                .is_err()
        };
        assert!(shrink(|b| b.x.bytes -= 1));
        assert!(shrink(|b| b.mixer.fn_.bytes -= 1));
        assert!(shrink(|b| b.mixer.scale.bytes -= 1));
        assert!(shrink(|b| b.comb.bytes -= 1));
        assert!(shrink(|b| b.scratch.tickets.as_mut().unwrap().bytes -= 1));
        assert!(shrink(|b| b.scratch.partials = None));
        assert!(shrink(|b| b.live_rows.bytes = 3));
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(0))
                .is_err()
        );
        assert!(
            prepared
                .launches(&buffers, View::EMPTY, Counts::rows(33))
                .is_err()
        );
        // The epilogue checks the all-reduce's views and its own.
        let prepared = backend
            .sinkhorn_reduce_expand_mix_norm(&reduce_query(4, 8), 0, Need::rows(8))
            .unwrap();
        let state = prepared.info().state.clone();
        let full = reduce_buffers(4, &state);
        prepared
            .launches(&full, View::EMPTY, Counts::rows(8))
            .unwrap();
        let mut missing = full.clone();
        missing.reduce.ranks.pop();
        assert!(
            prepared
                .launches(&missing, View::EMPTY, Counts::rows(8))
                .is_err()
        );
        let mut short = full.clone();
        short.norm_weight.bytes = 4096 * 2 - 1;
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(8))
                .is_err()
        );
        let mut short = full.clone();
        short.scratch.tickets = None;
        assert!(
            prepared
                .launches(&short, View::EMPTY, Counts::rows(8))
                .is_err()
        );
        assert!(
            prepared
                .launches(&full, View::EMPTY, Counts::rows(9))
                .is_err()
        );
    }

    #[test]
    fn the_split_epilogue_abi_is_the_one_bound() {
        let entry = sinkhorn_split::entry(PROFILE, SERVED.mix, eps(), 8).unwrap();
        let names: Vec<_> = entry
            .params
            .iter()
            .map(|param| match param {
                EntryParam::DevicePointer { name, .. } | EntryParam::U32 { name } => name.as_str(),
                _ => "",
            })
            .collect();
        assert_eq!(
            names,
            [
                "summed",
                "streams",
                "post",
                "comb",
                "reduced",
                "expanded",
                "fn",
                "base",
                "scale",
                "collapsed",
                "weight",
                "normed",
                "partials",
                "tickets",
                "live_rows",
                "eps_bits"
            ]
        );
    }

    /// The Sinkhorn keys hold what their entries launch, every field
    /// changed alone: symbol, ABI and code; never the profile or a
    /// capacity, which only bounds the rows. Each guard launches.
    #[test]
    fn launch_keys_hold_what_sinkhorn_entries_launch() {
        use crate::backend::select::{Class, Field, field_names, keys_follow_launches};
        let buffers = StreamExpandBuffers {
            x: view(1, u64::MAX),
            y: view(2, u64::MAX),
            post: view(3, u64::MAX),
            comb: view(4, u64::MAX),
            output: view(5, u64::MAX),
            live_rows: view(6, 4),
        };
        let base = expand_entries(&[(LEAST, 32)]).remove(0);
        let names = field_names!(sinkhorn_hc::Entry {
            profile,
            symbol,
            query,
            op,
            params
        });
        let fields: Vec<Field<sinkhorn_hc::Entry>> = vec![
            ("profile", Class::Inert, Box::new(|e| e.profile = WIDE)),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_sinkhorn_hc_other_v1".into()),
            ),
            (
                "query",
                Class::Inert,
                Box::new(|e| e.query = expand_query(64)),
            ),
            ("op", Class::Selects, Box::new(|e| e.op = Op::Head)),
            ("params", Class::Launch, Box::new(|e| e.params.swap(0, 1))),
        ];
        keys_follow_launches(&base, names, fields, one_cta_launch, |e| {
            let prepared = Sm120::new()
                .with_sinkhorn(vec![e.clone()])
                .sinkhorn_expand(&expand_query(64), Need::rows(8))?;
            let launches = [1, 8].map(|rows| {
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap()
            });
            Ok((format!("{} {:?}", e.body(), e.params), launches))
        });
        let split = split::entry(PROFILE, SERVED);
        let names = field_names!(split::Entry {
            profile,
            symbol,
            query,
            params
        });
        let fields: Vec<Field<split::Entry>> = vec![
            ("profile", Class::Inert, Box::new(|e| e.profile = WIDE)),
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_sinkhorn_hc_other_split_v1".into()),
            ),
            (
                "query",
                Class::Inert,
                Box::new(|e| e.query = mix_collapse(64)),
            ),
            ("params", Class::Launch, Box::new(|e| e.params.swap(0, 1))),
        ];
        keys_follow_launches(&split, names, fields, split_launch, |e| {
            let prepared = Sm120::new()
                .with_sinkhorn_split(e.clone())
                .sinkhorn_mix_collapse(&mix_collapse(32), Need::rows(8))?;
            let launches = [1, 8].map(|rows| {
                prepared
                    .launches(&mix_buffers(), View::EMPTY, Counts::rows(rows))
                    .unwrap()
            });
            Ok((format!("{:?} {:?}", e.body(), e.params), launches))
        });
        // The comm epilogue, after its all-reduce, at the rows it launches.
        let epilogue = sinkhorn_split::entry(PROFILE, SERVED.mix, eps(), SERVED.capacity).unwrap();
        let names = field_names!(sinkhorn_split::Entry {
            symbol,
            mix,
            rmsnorm_eps_bits,
            norm,
            capacity,
            params
        });
        let fields: Vec<Field<sinkhorn_split::Entry>> = vec![
            (
                "symbol",
                Class::Launch,
                Box::new(|e| e.symbol = "chop_comm_sinkhorn_other_v1".into()),
            ),
            ("mix", Class::Selects, Box::new(|e| e.mix.iterations += 1)),
            (
                "rmsnorm_eps_bits",
                Class::Selects,
                Box::new(|e| e.rmsnorm_eps_bits += 1),
            ),
            (
                "norm",
                Class::Selects,
                Box::new(|e| e.norm = RmsNormReduction::WarpTree { threads: 256 }),
            ),
            ("capacity", Class::Inert, Box::new(|e| e.capacity /= 2)),
            ("params", Class::Launch, Box::new(|e| e.params.swap(0, 1))),
        ];
        let (ranks, rank, rows) = (4, 1, 32);
        keys_follow_launches(&epilogue, names, fields, epilogue_launch, |e| {
            let prepared = Sm120::new()
                .with_sinkhorn_epilogue(e.clone())
                .sinkhorn_reduce_expand_mix_norm(
                    &reduce_query(ranks, rows),
                    rank,
                    Need::rows(rows),
                )?;
            let buffers = reduce_buffers(ranks, &prepared.info().state);
            let launches = [1, 2, rows].map(|count| {
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(count))
                    .unwrap()
            });
            Ok((format!("{} {:?}", e.body(), e.params), launches))
        });
    }
}
