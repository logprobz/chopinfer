//! The context-parallel invariance adapter: every `OpKind::ContextParallel`
//! entry, the entries of a model's context-parallel group of size 2
//! (`ModelCatalog::context_parallel`), on the group's own cubins.
//! Both group ranks run on one device: a launch takes its group rank and
//! size as scalars, and what a launch reads from its peer (gathered pairs,
//! partials) is built on the host, the bits the group transfers deliver
//! (the transfers' own claims are the comm adapter's). Every row's inputs
//! come from its seed (or its position) only, so neither a split nor the
//! other rows of a launch change them.
//!
//! - The position localization, partial merge, block-id merge, head-sum
//!   selection with its split merge and the selected partials read their
//!   rows and the cache or block keys of one sequence of `CONTEXT` tokens,
//!   placed on the group's pages once per case; a unit is the row's outputs
//!   on both group ranks.
//! - The owner-routed KV write: a unit is the row's Q on both ranks and its
//!   K and V at its owner's local page slot.
//! - The owner-routed key write keeps per-slot state: a case continues one
//!   history (written in launches that blocks straddle), and its units are
//!   each rank's ring and block keys after the case.
//!
//! A rows case is one sequence's exact verify step: R rows ending at each of
//! `VERIFY_ENDS`, launched together (the selected partials' verify entry),
//! against each row launched alone in order, as one-row decode steps (the
//! partials' decode entry, the verify entries' one-row reference: the same
//! instantiation, `verify_partials_share_the_decode_instantiation`). A
//! chunk case continues `PREFIX` cached tokens chunk by chunk. The anchor
//! case runs key writes that keep an anchor ring in a spare entry.
//!
//! Every launch is prepared for its served bucket, the power of two at or
//! above its live rows (`prepared`), with buffers of the prepared rows:
//! inputs past the live rows are poison, and outputs past them must keep
//! their fill. Every launch runs under the poison replays (eager,
//! relaunch, graph and PDL, `poison::replay_all`): outputs start poisoned,
//! or filled with `FILL` where all-ones words are values (-1 positions,
//! pools and block ids), write-only state is poisoned, and each is checked
//! against its write map after every launch; the selection's workspace may
//! hold only the live rows' lists.

use std::collections::BTreeMap;
use std::ops::Range;

use crate::invariance::{Adapter, Compared, Units};
use kernel_build::EntryParam;
use kernel_core::codec::f32_to_bf16;
use kernel_core::context_parallel_oracle::{owned_block_row, owned_kv_positions};
use kernel_core::contract::{AttentionPhase, DType, LaunchPlan};
use kernel_core::properties::{Anchors, Chunk};
use kernel_lab::poison::{POISON, Replay, WriteMap, replay_all};
use kernel_lab::{LabResult, Session, launch};
use kernels_sm120::context_parallel::{self as cp, GroupEntries, MergeLayout};
use kernels_sm120::invariance::{Declared, OpKind};
use kernels_sm120::owned_writes;
use kernels_sm120::pooled_indexer::{block_merge, head_sum};
use kernels_sm120::selected_partials;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Module};

#[path = "../common/selected_partials.rs"]
mod fixture;
use fixture::{BLOCK, DIM, Kv, PAGE, Q_HEADS, Rng, Step, TOP, download, rank_cache, upload};

const GROUP: usize = 2;
/// Tokens of every case's sequence: the chunk prompts' end.
const CONTEXT: usize = PREFIX + 512;
/// Cached tokens before a chunk case's prompt.
const PREFIX: usize = 4096;
/// Contexts a rows case's verify item ends at: the first rows that select
/// among more than `TOP` blocks (2,052 tokens), and a block completed
/// inside the item before a row on a new page (4,097).
const VERIFY_ENDS: [usize; 2] = [2052, 4097];
const INDEX_HEADS: usize = 4;
const INDEX_DIM: usize = 128;
/// An indexer projection row: the query heads, then the raw key.
const PROJECTION: usize = (INDEX_HEADS + 1) * INDEX_DIM;
/// A packed attention row: each query head beside its gate, then K and V.
const ROW: usize = Q_HEADS * 2 * DIM + 2 * DIM;
/// Pools of one page.
const PAGE_POOLS: usize = PAGE / BLOCK;
/// Block-key page stride: 4 KiB of keys and a 1 KiB gap.
const KEY_STRIDE: usize = 5120;
/// Bytes after each K and V page that a read or write outside it would hit.
const DECOY: usize = 512;
/// Rows of one history launch: not a multiple of the block, so blocks
/// straddle launches and the key write pools them from the ring.
const HISTORY_CHUNK: usize = 4093;
/// How far in after a launch's start the anchor case anchors for a plain
/// spare claim, which covers every anchor: inside the first ring of rows.
const NEAR_ANCHOR: usize = 4;
/// Spare anchor ring entries; the anchored launch names the last, so the
/// first must stay untouched.
const ANCHOR_SLOTS: usize = 2;
/// The fill of outputs whose all-ones words are values.
const FILL: [u8; 1] = [0x5a];
/// Kept tokens of a selection: every block the entries keep.
const TOP_K: u32 = (TOP * BLOCK) as u32;
const SOFTMAX_SCALE: f32 = 0.0625;

pub struct ContextParallel;

/// The entry of a declared symbol, by kind.
#[derive(Clone, Debug)]
enum Stage {
    Localize(cp::Entry),
    Merge(cp::Entry),
    Blocks(block_merge::Entry),
    /// The selection and its split merge, for either symbol.
    Select(head_sum::Entry),
    Partials(selected_partials::Entry),
    KvWrite(owned_writes::Entry),
    KeyWrite(owned_writes::Entry),
}

/// Every model's context-parallel groups, in catalog order.
fn groups() -> Vec<GroupEntries> {
    kernel_lab::catalogs()
        .models()
        .iter()
        .flat_map(|model| (model.context_parallel)())
        .collect()
}

/// The group that holds `symbol`.
fn group_of(symbol: &str) -> LabResult<GroupEntries> {
    groups()
        .into_iter()
        .find(|group| group.symbols().iter().any(|s| s == symbol))
        .ok_or_else(|| format!("{symbol}: no context-parallel entry"))
}

/// The cubin of the group that holds `symbol`.
fn cubin_of(symbol: &str) -> LabResult<String> {
    groups()
        .into_iter()
        .flat_map(|group| group.cubins)
        .find(|cubin| cubin.entries.iter().any(|entry| entry.symbol == symbol))
        .map(|cubin| cubin.name)
        .ok_or_else(|| format!("{symbol}: no context-parallel cubin"))
}

fn stage_of(symbol: &str) -> LabResult<Stage> {
    let group = group_of(symbol)?;
    if let Some(entry) = group.group.iter().find(|entry| entry.symbol() == symbol) {
        return Ok(match entry.kind {
            cp::Kind::LocalizeKeys { .. } => Stage::Localize(entry.clone()),
            cp::Kind::PartialMerge { .. } => Stage::Merge(entry.clone()),
        });
    }
    if symbol == group.block_merge.symbol() {
        return Ok(Stage::Blocks(group.block_merge));
    }
    let select = group.select;
    if symbol == select.symbol() || select.merge_symbol().as_deref() == Some(symbol) {
        return Ok(Stage::Select(select));
    }
    if let Some(entry) = group
        .partials
        .into_iter()
        .find(|entry| entry.symbol() == symbol)
    {
        return Ok(Stage::Partials(entry));
    }
    if let Some(entry) = group
        .owned
        .into_iter()
        .find(|entry| entry.symbol() == symbol)
    {
        return Ok(match entry.kind {
            owned_writes::Kind::KvWrite { .. } => Stage::KvWrite(entry),
            owned_writes::Kind::KeyWrite => Stage::KeyWrite(entry),
        });
    }
    Err(format!("{symbol}: no context-parallel entry"))
}

/// The partials entry a row launched alone takes: the decode entry of the
/// verify entry's KV dtype, the verify entries' one-row reference.
fn one_row_partials(entry: &selected_partials::Entry) -> LabResult<selected_partials::Entry> {
    if entry.phase != AttentionPhase::Verify {
        return Ok(entry.clone());
    }
    group_of(&entry.symbol())?
        .partials
        .into_iter()
        .find(|e| e.phase == AttentionPhase::Decode && e.kv == entry.kv)
        .ok_or_else(|| format!("{}: no decode entry", entry.symbol()))
}

/// The fixtures' geometry is the rank geometry of `symbol`'s group: six
/// Q heads over one KV head of 256 in pages of 64, and a four-head indexer
/// of 128 in blocks of 4 keeping 512 blocks.
fn check_geometry(symbol: &str) -> LabResult<()> {
    let group = group_of(symbol)?;
    let s = group.select.geometry;
    let key = group
        .owned
        .iter()
        .find(|entry| entry.kind == owned_writes::Kind::KeyWrite)
        .ok_or("a group without an owner-routed key write")?;
    let same = group.ranks == GROUP as u32
        && group.partials.iter().all(|entry| {
            let g = entry.geometry;
            (
                g.q_heads * g.kv_heads,
                g.head_dim,
                g.page,
                g.block,
                g.top_blocks,
            ) == (
                Q_HEADS as u32,
                DIM as u32,
                PAGE as u32,
                BLOCK as u32,
                TOP as u32,
            )
        })
        && (s.heads, s.head_dim, s.group, s.max_top_k)
            == (INDEX_HEADS as u32, INDEX_DIM as u32, BLOCK as u32, TOP_K)
        && key.indexer.softmax_scale == SOFTMAX_SCALE
        && key.indexer.rotary_dim == key.gated.rotary
        && key.indexer.theta == key.gated.theta
        && key.indexer.mrope[1..] == key.gated.sections[1..];
    if !same {
        return Err(format!(
            "{symbol}: the fixtures' geometry is not its group's"
        ));
    }
    Ok(())
}

/// A generator for the parts of one input (splitmix64 of the parts).
fn rng(parts: &[u64]) -> Rng {
    let mut z = 0x9e37_79b9_7f4a_7c15_u64;
    for &part in parts {
        z ^= part;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
    }
    Rng(z | 1)
}

fn bf16s(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|v| f32_to_bf16(v).to_ne_bytes())
        .collect()
}

fn words(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values.into_iter().flat_map(i32::to_ne_bytes).collect()
}

fn floats(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_ne_bytes).collect()
}

fn refusal(e: kernel_core::contract::Refusal) -> String {
    e.0.to_string()
}

/// A launch argument, bound to an entry parameter by name.
enum Value<'a> {
    Buffer(&'a DeviceBuffer),
    U32(u32),
    U64(u64),
}

/// `values` in the order of `params`, each parameter bound once.
fn arguments<'a>(
    params: &[EntryParam],
    values: Vec<(&str, Value<'a>)>,
) -> LabResult<Vec<KernelArgument<'a>>> {
    if params.len() != values.len() {
        return Err("a launch binds a parameter twice or not at all".into());
    }
    let mut values = values;
    params
        .iter()
        .map(|param| {
            let name = match param {
                EntryParam::DevicePointer { name, .. }
                | EntryParam::U32 { name }
                | EntryParam::U64 { name } => name,
                other => return Err(format!("unexpected parameter {other:?}")),
            };
            let at = values
                .iter()
                .position(|(bound, _)| bound == name)
                .ok_or_else(|| format!("no value for {name}"))?;
            Ok(match (param, values.remove(at).1) {
                (EntryParam::DevicePointer { .. }, Value::Buffer(buffer)) => {
                    KernelArgument::Buffer(buffer)
                }
                (EntryParam::U32 { .. }, Value::U32(value)) => KernelArgument::U32(value),
                (EntryParam::U64 { .. }, Value::U64(value)) => KernelArgument::U64(value),
                _ => return Err(format!("{name} has the wrong kind")),
            })
        })
        .collect()
}

fn function(module: &Module, symbol: &str, plan: LaunchPlan) -> LabResult<Function> {
    let function = module.function(symbol).map_err(|e| e.to_string())?;
    if plan.dynamic_shared_bytes > 48 * 1024 {
        function
            .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
    }
    Ok(function)
}

/// One kernel of a replayed launch: its symbol, function, plan and
/// arguments.
type Kernel<'a> = (String, &'a Function, LaunchPlan, Vec<KernelArgument<'a>>);

/// Runs `kernels` in order under the poison replays of `replay`, on the
/// live count in `live`.
fn replayed(
    session: &Session,
    replay: Replay<'_>,
    live: &DeviceBuffer,
    kernels: &[Kernel<'_>],
) -> LabResult<()> {
    let run = || {
        for (symbol, function, plan, args) in kernels {
            function
                .launch(&session.stream, launch::config(*plan)?, args)
                .map_err(|e| e.to_string())?;
            kernel_lab::launched(symbol);
        }
        Ok(())
    };
    replay_all(session, replay, live, run, &mut BTreeMap::new())
}

/// A rank's page placement: physical page j lives on group rank `j % 2` at
/// local page `table[j / 2]`; local pages run opposite to the stripes.
fn table(pages: usize) -> Vec<i32> {
    (0..pages.div_ceil(GROUP) as i32).rev().collect()
}

/// The group rank, local page and slot of position `t`.
fn place(table: &[i32], t: usize) -> (usize, usize, usize) {
    let physical = t / PAGE;
    (physical % GROUP, table[physical / GROUP] as usize, t % PAGE)
}

/// One case: a sequence's rows (positions and seeds) and its launches,
/// each a range of the rows.
struct Case {
    positions: Vec<usize>,
    seeds: Vec<u64>,
    launches: Vec<Range<usize>>,
}

impl Case {
    /// A verify item of `seeds.len()` rows ending at `end`, in one launch.
    fn verify(end: usize, seeds: &[u64]) -> Case {
        let rows = seeds.len();
        Case {
            positions: (end - rows..end).collect(),
            seeds: seeds.to_vec(),
            launches: std::iter::once(0..rows).collect(),
        }
    }

    /// The same rows, each launched alone in order.
    fn alone(end: usize, seeds: &[u64]) -> Case {
        let mut case = Case::verify(end, seeds);
        case.launches = (0..seeds.len()).map(|row| row..row + 1).collect();
        case
    }

    /// A prompt of `rows` rows after `PREFIX` cached tokens, in chunks
    /// ending at `ends`; row r's seed is r.
    fn chunks(rows: u32, ends: &[u32]) -> Case {
        let mut start = 0;
        Case {
            positions: (PREFIX..PREFIX + rows as usize).collect(),
            seeds: (0..u64::from(rows)).collect(),
            launches: ends
                .iter()
                .map(|&end| {
                    let range = start..end as usize;
                    start = end as usize;
                    range
                })
                .collect(),
        }
    }

    fn rows(&self) -> usize {
        self.positions.len()
    }
}

/// Each row's units: `parts[launch][rank]` holds a launch's rows of
/// `width` bytes each, the units the row's part on every rank in order.
fn units(case: &Case, parts: &[Vec<Vec<u8>>], width: usize) -> Units {
    let mut units = vec![Vec::new(); case.rows()];
    for (launch, ranks) in case.launches.iter().zip(parts) {
        for bytes in ranks {
            for (i, row) in launch.clone().enumerate() {
                units[row].extend_from_slice(&bytes[i * width..(i + 1) * width]);
            }
        }
    }
    units
}

/// The rows a launch of `n` live rows is prepared for: the served bucket,
/// the power of two at or above them. Every buffer holds the prepared rows;
/// inputs past the live rows are poison, and outputs past them must keep
/// their fill.
fn prepared(n: usize) -> usize {
    n.next_power_of_two()
}

/// The equal rows of `n` live rows, padded with poison to the prepared
/// rows.
fn padded(mut bytes: Vec<u8>, n: usize) -> Vec<u8> {
    let row = bytes.len() / n;
    assert_eq!(bytes.len(), n * row, "equal rows");
    bytes.resize(prepared(n) * row, POISON);
    bytes
}

/// The localized positions of every rank.
fn localize(session: &Session, entry: &cp::Entry, case: &Case) -> LabResult<Units> {
    let module = session.module(&cubin_of(&entry.symbol())?)?;
    let mut parts = Vec::new();
    for launch in &case.launches {
        let n = launch.len();
        let plan = entry.plan(prepared(n) as u32).map_err(refusal)?;
        let lists = upload(
            session,
            &padded(
                words(case.positions[launch.clone()].iter().map(|&p| p as i32)),
                n,
            ),
        )?;
        let live = upload(session, &(n as u32).to_ne_bytes())?;
        let mut ranks = Vec::new();
        for rank in 0..GROUP {
            entry.check_rank(rank as u32).map_err(refusal)?;
            let out = session.allocate(prepared(n) * 4)?;
            let args = arguments(
                &entry.params(),
                vec![
                    ("lists", Value::Buffer(&lists)),
                    ("live_rows", Value::Buffer(&live)),
                    ("out", Value::Buffer(&out)),
                    ("cp_rank", Value::U32(rank as u32)),
                ],
            )?;
            let replay = Replay::new(session).output_with(
                "out",
                &out,
                WriteMap::of(DType::I32).prefix(n)?,
                &FILL,
            );
            let kernel = function(&module, &entry.symbol(), plan)?;
            replayed(
                session,
                replay,
                &live,
                &[(entry.symbol(), &kernel, plan, args)],
            )?;
            ranks.push(download(session, &out, n * 4)?);
        }
        parts.push(ranks);
    }
    Ok(units(case, &parts, 4))
}

/// Row `seed`'s partials of every rank for the merge: `m` and `l` of each
/// head, then `o`. A rank is empty (`m = -inf`, `l = o = 0`) for some
/// rows, and both are for a few.
fn merge_row(seed: u64) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    (0..GROUP)
        .map(|rank| {
            let mut rng = rng(&[seed, rank as u64, 0x6d]);
            let empty = (rank == 1 && seed % 8 == 3) || seed % 16 == 5;
            let mut m = Vec::new();
            let mut l = Vec::new();
            for _ in 0..Q_HEADS {
                m.push(if empty {
                    f32::NEG_INFINITY
                } else {
                    8.0 * rng.unit()
                });
                l.push(if empty {
                    0.0
                } else {
                    0.5 + 4.0 * rng.unit().abs()
                });
            }
            let o = (0..Q_HEADS * DIM)
                .map(|_| if empty { 0.0 } else { 4.0 * rng.unit() })
                .collect();
            (m, l, o)
        })
        .collect()
}

/// The merged attention rows: each launch merges its rows' partials,
/// gathered in the exchanged layout of the launch's bucket (the rows past
/// the live ones NaN).
fn merge(session: &Session, entry: &cp::Entry, case: &Case) -> LabResult<Units> {
    let module = session.module(&cubin_of(&entry.symbol())?)?;
    let heads = Q_HEADS;
    let mut parts = Vec::new();
    for launch in &case.launches {
        let n = launch.len();
        let bucket = prepared(n);
        let layout = MergeLayout::exchanged(bucket as u32, heads as u32);
        entry.check_merge(n as u32, layout).map_err(refusal)?;
        let rows = case.seeds[launch.clone()]
            .iter()
            .map(|&seed| merge_row(seed))
            .collect::<Vec<_>>();
        let mut segments = Vec::new();
        for rank in 0..GROUP {
            let mut m = vec![f32::NAN; bucket * heads];
            let mut l = vec![f32::NAN; bucket * heads];
            let mut o = vec![f32::NAN; bucket * heads * DIM];
            for (row, parts) in rows.iter().enumerate() {
                let (rm, rl, ro) = &parts[rank];
                m[row * heads..][..heads].copy_from_slice(rm);
                l[row * heads..][..heads].copy_from_slice(rl);
                o[row * heads * DIM..][..heads * DIM].copy_from_slice(ro);
            }
            segments.extend(floats(m.into_iter().chain(l).chain(o)));
        }
        let partials = upload(session, &segments)?;
        let live = upload(session, &(n as u32).to_ne_bytes())?;
        let out = session.allocate(bucket * heads * DIM * 2)?;
        let plan = entry.plan(bucket as u32).map_err(refusal)?;
        let args = arguments(
            &entry.params(),
            vec![
                ("partials", Value::Buffer(&partials)),
                ("live_rows", Value::Buffer(&live)),
                ("out", Value::Buffer(&out)),
                ("rank_rows", Value::U32(layout.rank_rows)),
                ("input_heads", Value::U32(layout.input_heads)),
                ("head_offset", Value::U32(layout.head_offset)),
            ],
        )?;
        let replay = Replay::new(session).output(
            "out",
            &out,
            WriteMap::of(DType::Bf16).prefix(n * heads * DIM)?,
        );
        let kernel = function(&module, &entry.symbol(), plan)?;
        replayed(
            session,
            replay,
            &live,
            &[(entry.symbol(), &kernel, plan, args)],
        )?;
        parts.push(vec![download(session, &out, n * heads * DIM * 2)?]);
    }
    Ok(units(case, &parts, heads * DIM * 2))
}

/// Rank `rank`'s (score, pool) pairs of a row at `position`, best first:
/// coarse scores (so ties occur) of the complete pools the rank owns,
/// padded with (-inf, -1).
fn rank_pairs(seed: u64, position: usize, rank: usize) -> Vec<u32> {
    let mut rng = rng(&[seed, position as u64, rank as u64, 0x70]);
    let complete = (position + 1) / BLOCK;
    let mut keyed = (0..complete)
        .filter(|&b| (b / PAGE_POOLS) % GROUP == rank)
        .map(|b| ((rng.next() % 64) as f32 / 8.0 - 4.0, b as u32))
        .collect::<Vec<_>>();
    keyed.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    keyed.truncate(TOP);
    let mut pairs = keyed
        .into_iter()
        .flat_map(|(score, pool)| [score.to_bits(), pool])
        .collect::<Vec<_>>();
    pairs.resize(TOP * 2, 0);
    for pad in pairs
        .chunks_exact_mut(2)
        .skip_while(|p| p[1] != 0 || p[0] != 0)
    {
        pad.copy_from_slice(&[f32::NEG_INFINITY.to_bits(), u32::MAX]);
    }
    pairs
}

/// The block ids each launch merges from its rows' pairs, gathered
/// rank-major at the launch's bucket (the rows past the live ones all
/// ones).
fn blocks(session: &Session, entry: &block_merge::Entry, case: &Case) -> LabResult<Units> {
    let module = session.module(&cubin_of(&entry.symbol())?)?;
    let mut parts = Vec::new();
    for launch in &case.launches {
        let n = launch.len();
        let bucket = prepared(n);
        let mut gathered = vec![u32::MAX; GROUP * bucket * TOP * 2];
        for rank in 0..GROUP {
            for (i, row) in launch.clone().enumerate() {
                let pairs = rank_pairs(case.seeds[row], case.positions[row], rank);
                gathered[(rank * bucket + i) * TOP * 2..][..TOP * 2].copy_from_slice(&pairs);
            }
        }
        let pairs = upload(
            session,
            &gathered
                .iter()
                .flat_map(|w| w.to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        let live = upload(session, &(n as u32).to_ne_bytes())?;
        let out = session.allocate(bucket * TOP * 4)?;
        let plan = entry.plan(bucket as u32).map_err(refusal)?;
        let args = arguments(
            &entry.params(),
            vec![
                ("pairs", Value::Buffer(&pairs)),
                ("live_rows", Value::Buffer(&live)),
                ("blocks", Value::Buffer(&out)),
                ("top_k", Value::U32(TOP_K)),
                ("rank_rows", Value::U32(bucket as u32)),
            ],
        )?;
        let replay = Replay::new(session).output_with(
            "blocks",
            &out,
            WriteMap::of(DType::I32).prefix(n * TOP)?,
            &FILL,
        );
        let kernel = function(&module, &entry.symbol(), plan)?;
        replayed(
            session,
            replay,
            &live,
            &[(entry.symbol(), &kernel, plan, args)],
        )?;
        parts.push(vec![download(session, &out, n * TOP * 4)?]);
    }
    Ok(units(case, &parts, TOP * 4))
}

/// Each rank's block keys of a `CONTEXT`-token sequence: pool b's key in
/// row `b % PAGE_POOLS` of its local page, other bytes a BF16 decoy.
fn block_keys(seed: u64) -> (Vec<i32>, Vec<Vec<u8>>) {
    let pages = CONTEXT.div_ceil(PAGE);
    let table = table(pages);
    let mut keys = vec![vec![0x7f_u8; table.len() * KEY_STRIDE]; GROUP];
    for b in 0..CONTEXT / BLOCK {
        let (rank, local, _) = place(&table, b * BLOCK);
        let mut rng = rng(&[seed, b as u64, 0x6b]);
        let at = local * KEY_STRIDE + (b % PAGE_POOLS) * INDEX_DIM * 2;
        keys[rank][at..at + INDEX_DIM * 2]
            .copy_from_slice(&bf16s((0..INDEX_DIM).map(|_| rng.unit())));
    }
    (table, keys)
}

/// Each rank's selected pairs of every row: the per-rank selection over the
/// rank's own block keys and its split merge, in one replayed launch.
fn select(session: &Session, entry: &head_sum::Entry, case: &Case) -> LabResult<Units> {
    let module = session.module(&cubin_of(&entry.symbol())?)?;
    let merge_symbol = entry.merge_symbol().ok_or("a selection without a merge")?;
    entry.check_top_k(TOP_K).map_err(refusal)?;
    let (table, keys) = block_keys(0xb10c);
    let tables = upload(session, &words(table.iter().copied()))?;
    let keys = keys
        .iter()
        .map(|bytes| upload(session, bytes))
        .collect::<LabResult<Vec<_>>>()?;
    let mut parts = Vec::new();
    for launch in &case.launches {
        let n = launch.len();
        let bucket = prepared(n);
        let q = upload(
            session,
            &padded(
                bf16s(case.seeds[launch.clone()].iter().flat_map(|&seed| {
                    let mut rng = rng(&[seed, 0x71]);
                    (0..INDEX_HEADS * INDEX_DIM).map(move |_| 2.0 * rng.unit())
                })),
                n,
            ),
        )?;
        let row_batch = upload(session, &padded(words(vec![0; n]), n))?;
        let positions = upload(
            session,
            &padded(
                words(case.positions[launch.clone()].iter().map(|&p| p as i32)),
                n,
            ),
        )?;
        let live = upload(session, &(n as u32).to_ne_bytes())?;
        let splits = entry.splits(bucket as u32);
        let plan = entry.plan(bucket as u32).map_err(refusal)?;
        let merge_plan = entry
            .merge_plan(bucket as u32)
            .map_err(refusal)?
            .ok_or("a selection without a merge plan")?;
        // The workspace is `[rows][splits][kept]`: only the live rows'
        // lists may be written.
        let workspace_bytes = entry.workspace_bytes(bucket as u32) as usize;
        let live_workspace = workspace_bytes / bucket * n;
        let mut ranks = Vec::new();
        for (rank, keys) in keys.iter().enumerate() {
            entry
                .check_context_parallel(rank as u32, GROUP as u32)
                .map_err(refusal)?;
            let workspace = session.allocate(workspace_bytes)?;
            let out = session.allocate(bucket * TOP * 2 * 4)?;
            let select = arguments(
                &entry.params(),
                vec![
                    ("q", Value::Buffer(&q)),
                    ("pooled_keys", Value::Buffer(keys)),
                    ("page_tables", Value::Buffer(&tables)),
                    ("row_batch", Value::Buffer(&row_batch)),
                    ("positions", Value::Buffer(&positions)),
                    ("live_rows", Value::Buffer(&live)),
                    ("workspace", Value::Buffer(&workspace)),
                    ("max_pages", Value::U32(table.len() as u32)),
                    ("page_stride_bytes", Value::U64(KEY_STRIDE as u64)),
                    ("top_k", Value::U32(TOP_K)),
                    ("splits", Value::U32(splits)),
                    ("cp_rank", Value::U32(rank as u32)),
                    ("cp_size", Value::U32(GROUP as u32)),
                ],
            )?;
            let merge = arguments(
                &entry.merge_params(),
                vec![
                    ("workspace", Value::Buffer(&workspace)),
                    ("positions", Value::Buffer(&positions)),
                    ("live_rows", Value::Buffer(&live)),
                    ("output", Value::Buffer(&out)),
                    ("top_k", Value::U32(TOP_K)),
                    ("splits", Value::U32(splits)),
                ],
            )?;
            let replay = Replay::new(session)
                .scratch_within(
                    "workspace",
                    &workspace,
                    std::iter::once(0..live_workspace).collect(),
                )
                .output_with(
                    "pairs",
                    &out,
                    WriteMap::of(DType::I32).prefix(n * TOP * 2)?,
                    &FILL,
                );
            let (selects, merges) = (
                function(&module, &entry.symbol(), plan)?,
                function(&module, &merge_symbol, merge_plan)?,
            );
            replayed(
                session,
                replay,
                &live,
                &[
                    (entry.symbol(), &selects, plan, select),
                    (merge_symbol.clone(), &merges, merge_plan, merge),
                ],
            )?;
            ranks.push(download(session, &out, n * TOP * 2 * 4)?);
        }
        parts.push(ranks);
    }
    Ok(units(case, &parts, TOP * 2 * 4))
}

/// Each rank's partials of every row over its own pages of one sequence:
/// `m`, `l` and `o` of the row's heads.
fn partials(session: &Session, entry: &selected_partials::Entry, case: &Case) -> LabResult<Units> {
    let module = session.module(&cubin_of(&entry.symbol())?)?;
    let fp8 = entry.kv == DType::Fp8E4M3;
    let kv = Kv::new(&[CONTEXT], fp8, false, &mut rng(&[0x6b76, u64::from(fp8)]));
    let caches = (0..GROUP)
        .map(|rank| rank_cache(&kv, rank, GROUP, false, &mut rng(&[0xcac4e, rank as u64])))
        .collect::<Vec<_>>();
    let (_, k_scale, v_scale) = fixture::scales(fp8);
    let heads = Q_HEADS;
    let row_floats = heads * (DIM + 2);
    let mut parts = Vec::new();
    for launch in &case.launches {
        let n = launch.len();
        let bucket = prepared(n);
        let rows = launch.clone();
        let q = upload(
            session,
            &padded(
                rows.clone()
                    .flat_map(|row| fixture::queries(1, false, &mut rng(&[case.seeds[row], 0x71])))
                    .flat_map(|b| b.to_ne_bytes())
                    .collect::<Vec<_>>(),
                n,
            ),
        )?;
        let selected = upload(
            session,
            &padded(
                words(rows.clone().flat_map(|row| {
                    let step = Step {
                        lengths: vec![CONTEXT],
                        rows: vec![(0, case.positions[row])],
                    };
                    fixture::selections(&step, &mut rng(&[case.seeds[row], 0x73]))
                })),
                n,
            ),
        )?;
        let row_batch = upload(session, &padded(words(vec![0; n]), n))?;
        let positions = upload(
            session,
            &padded(words(rows.clone().map(|row| case.positions[row] as i32)), n),
        )?;
        let live = upload(session, &(n as u32).to_ne_bytes())?;
        let plan = entry.plan(bucket as u32).map_err(refusal)?;
        let mut ranks = Vec::new();
        for (rank, cache) in caches.iter().enumerate() {
            entry
                .check_rank(rank as u32, GROUP as u32)
                .map_err(refusal)?;
            let k = upload(session, &cache.k)?;
            let v = upload(session, &cache.v)?;
            let tables = upload(session, &fixture::words(&cache.table_words()))?;
            let out_max = session.allocate(bucket * heads * 4)?;
            let out_den = session.allocate(bucket * heads * 4)?;
            let out_num = session.allocate(bucket * heads * DIM * 4)?;
            let args = arguments(
                &entry.params(),
                vec![
                    ("q", Value::Buffer(&q)),
                    ("k_cache", Value::Buffer(&k)),
                    ("v_cache", Value::Buffer(&v)),
                    ("block_tables", Value::Buffer(&tables)),
                    ("row_batch", Value::Buffer(&row_batch)),
                    ("positions", Value::Buffer(&positions)),
                    ("live_rows", Value::Buffer(&live)),
                    ("selected", Value::Buffer(&selected)),
                    ("out_max", Value::Buffer(&out_max)),
                    ("out_den", Value::Buffer(&out_den)),
                    ("out_num", Value::Buffer(&out_num)),
                    ("max_pages", Value::U32(cache.max_pages as u32)),
                    ("k_page_stride_bytes", Value::U64(cache.stride as u64)),
                    ("v_page_stride_bytes", Value::U64(cache.stride as u64)),
                    ("scale_bits", Value::U32(SOFTMAX_SCALE.to_bits())),
                    ("k_scale_bits", Value::U32(k_scale.to_bits())),
                    ("v_scale_bits", Value::U32(v_scale.to_bits())),
                    ("cp_rank", Value::U32(rank as u32)),
                    ("cp_size", Value::U32(GROUP as u32)),
                ],
            )?;
            let replay = Replay::new(session)
                .output(
                    "maximum",
                    &out_max,
                    WriteMap::of(DType::F32).prefix(n * heads)?,
                )
                .output(
                    "denominator",
                    &out_den,
                    WriteMap::of(DType::F32).prefix(n * heads)?,
                )
                .output(
                    "numerator",
                    &out_num,
                    WriteMap::of(DType::F32).prefix(n * heads * DIM)?,
                );
            let kernel = function(&module, &entry.symbol(), plan)?;
            replayed(
                session,
                replay,
                &live,
                &[(entry.symbol(), &kernel, plan, args)],
            )?;
            // Row-major per row: its maxima, denominators and numerators.
            let (m, l, o) = (
                download(session, &out_max, n * heads * 4)?,
                download(session, &out_den, n * heads * 4)?,
                download(session, &out_num, n * heads * DIM * 4)?,
            );
            let mut bytes = Vec::with_capacity(n * row_floats * 4);
            for row in 0..n {
                bytes.extend_from_slice(&m[row * heads * 4..][..heads * 4]);
                bytes.extend_from_slice(&l[row * heads * 4..][..heads * 4]);
                bytes.extend_from_slice(&o[row * heads * DIM * 4..][..heads * DIM * 4]);
            }
            ranks.push(bytes);
        }
        parts.push(ranks);
    }
    Ok(units(case, &parts, row_floats * 4))
}

/// The F32 cos/sin table `[position][pair][cos, sin]` of the writes'
/// M-RoPE, for text positions (all three sections at the token's index).
fn cos_sin(entry: &owned_writes::Entry) -> Vec<u8> {
    let g = entry.gated;
    floats((0..CONTEXT).flat_map(|position| {
        kernel_core::gated_attention_oracle::mrope_cos_sin(
            [position; 3],
            g.rotary as usize,
            f64::from(g.theta),
            g.sections,
        )
        .into_iter()
        .flat_map(|(cos, sin)| [cos as f32, sin as f32])
    }))
}

/// The K/V write map of a launch of positions `rows` on group rank `rank`:
/// each owned row's whole elements in its plane of pages at `stride`.
fn kv_map(
    table: &[i32],
    rank: usize,
    stride: usize,
    fp8: bool,
    rows: &[usize],
) -> LabResult<WriteMap> {
    let cache = WriteMap::of(if fp8 { DType::Fp8E4M3 } else { DType::Bf16 });
    let element = cache.bytes()?;
    cache.byte_ranges(rows.iter().filter_map(|&t| {
        let (owner, local, slot) = place(table, t);
        (owner == rank).then(|| {
            let at = local * stride + slot * DIM * element;
            at..at + DIM * element
        })
    }))
}

/// The owner-routed K/V write of every row: each rank's Q row, then the
/// row's K and V at its owner's slot.
fn kv_write(session: &Session, entry: &owned_writes::Entry, case: &Case) -> LabResult<Units> {
    let module = session.module(&cubin_of(&entry.symbol())?)?;
    let owned_writes::Kind::KvWrite { kv } = entry.kind else {
        return Err("not a K/V write".into());
    };
    let fp8 = kv == DType::Fp8E4M3;
    let element = if fp8 { 1 } else { 2 };
    let stride = PAGE * DIM * element + DECOY;
    let table = table(CONTEXT.div_ceil(PAGE));
    let decoy = if fp8 { 0x7e } else { 0x7f };
    let planes = (0..GROUP)
        .map(|_| {
            Ok([
                upload(session, &vec![decoy; table.len() * stride])?,
                upload(session, &vec![decoy; table.len() * stride])?,
            ])
        })
        .collect::<LabResult<Vec<_>>>()?;
    let mut weights = rng(&[0x6e6f726d]);
    let q_norm = upload(session, &bf16s((0..DIM).map(|_| 0.25 * weights.unit())))?;
    let k_norm = upload(session, &bf16s((0..DIM).map(|_| 0.25 * weights.unit())))?;
    let cos_sin = upload(session, &cos_sin(entry))?;
    let tables = upload(session, &words(table.iter().copied()))?;
    let q_row = Q_HEADS * DIM * 2;
    let mut parts = Vec::new();
    for launch in &case.launches {
        let n = launch.len();
        let rows = &case.positions[launch.clone()];
        let rope = upload(
            session,
            &padded(words(rows.iter().flat_map(|&p| [p as i32; 3])), n),
        )?;
        let row_batch = upload(session, &padded(words(vec![0; n]), n))?;
        let live = upload(session, &(n as u32).to_ne_bytes())?;
        let plan = entry.plan(prepared(n) as u32).map_err(refusal)?;
        let mut ranks = Vec::new();
        for (rank, [k, v]) in planes.iter().enumerate() {
            // The rank's own Q heads; K and V are the replica set's.
            let qkv = upload(
                session,
                &padded(
                    bf16s(case.seeds[launch.clone()].iter().flat_map(|&seed| {
                        let mut q = rng(&[seed, rank as u64, 0x71]);
                        let mut kv = rng(&[seed, 0x6b76]);
                        (0..ROW).map(move |i| {
                            if i >= Q_HEADS * 2 * DIM {
                                kv.unit()
                            } else if i % (2 * DIM) < DIM {
                                2.0 * q.unit()
                            } else {
                                q.unit()
                            }
                        })
                    })),
                    n,
                ),
            )?;
            let positions = rows.iter().map(|&p| p as i32).collect::<Vec<_>>();
            let local =
                owned_kv_positions(entry.query(), rank as u32, &positions).map_err(refusal)?;
            let local = upload(session, &padded(words(local), n))?;
            let q = session.allocate(prepared(n) * q_row)?;
            let args = arguments(
                &entry.params(),
                vec![
                    ("qkv", Value::Buffer(&qkv)),
                    ("q_norm", Value::Buffer(&q_norm)),
                    ("k_norm", Value::Buffer(&k_norm)),
                    ("rope_positions", Value::Buffer(&rope)),
                    ("cos_sin", Value::Buffer(&cos_sin)),
                    ("row_batch", Value::Buffer(&row_batch)),
                    ("local_positions", Value::Buffer(&local)),
                    ("live_rows", Value::Buffer(&live)),
                    ("block_tables", Value::Buffer(&tables)),
                    ("k_cache", Value::Buffer(k)),
                    ("v_cache", Value::Buffer(v)),
                    ("q", Value::Buffer(&q)),
                    ("max_pages", Value::U32(table.len() as u32)),
                    ("k_page_stride_bytes", Value::U64(stride as u64)),
                    ("v_page_stride_bytes", Value::U64(stride as u64)),
                    ("eps_bits", Value::U32(entry.gated.eps.to_bits())),
                    ("k_scale_bits", Value::U32(1.0_f32.to_bits())),
                    ("v_scale_bits", Value::U32(1.0_f32.to_bits())),
                ],
            )?;
            let map = kv_map(&table, rank, stride, fp8, rows)?;
            let replay = Replay::new(session)
                .output(
                    "q",
                    &q,
                    WriteMap::of(DType::Bf16).prefix(n * Q_HEADS * DIM)?,
                )
                .state(
                    "k_cache",
                    k,
                    download(session, k, k.byte_len())?,
                    map.clone(),
                    Vec::new(),
                )
                .state(
                    "v_cache",
                    v,
                    download(session, v, v.byte_len())?,
                    map,
                    Vec::new(),
                );
            let kernel = function(&module, &entry.symbol(), plan)?;
            replayed(
                session,
                replay,
                &live,
                &[(entry.symbol(), &kernel, plan, args)],
            )?;
            ranks.push(download(session, &q, n * q_row)?);
        }
        parts.push(ranks);
    }
    let mut units = units(case, &parts, q_row);
    // Each row's K and V at its owner's slot, after every launch.
    let images = planes
        .iter()
        .map(|[k, v]| {
            Ok([
                download(session, k, k.byte_len())?,
                download(session, v, v.byte_len())?,
            ])
        })
        .collect::<LabResult<Vec<_>>>()?;
    for (unit, &t) in units.iter_mut().zip(&case.positions) {
        let (owner, local, slot) = place(&table, t);
        for plane in &images[owner] {
            unit.extend_from_slice(
                &plane[local * stride + slot * DIM * element..][..DIM * element],
            );
        }
    }
    Ok(units)
}

/// One group rank's key-write state: its block keys, ring and anchor rings.
struct KeyState {
    keys: DeviceBuffer,
    ring_keys: DeviceBuffer,
    ring_positions: DeviceBuffer,
    anchor_keys: DeviceBuffer,
    anchor_positions: DeviceBuffer,
}

impl KeyState {
    fn new(session: &Session, table: &[i32], ring: usize) -> LabResult<KeyState> {
        Ok(KeyState {
            keys: upload(session, &vec![0x7f; table.len() * KEY_STRIDE])?,
            ring_keys: upload(session, &vec![0xff; ring * INDEX_DIM * 2])?,
            ring_positions: upload(session, &vec![0xff; ring * 3 * 4])?,
            anchor_keys: upload(session, &vec![0xff; ANCHOR_SLOTS * ring * INDEX_DIM * 2])?,
            anchor_positions: upload(session, &vec![0xff; ANCHOR_SLOTS * ring * 3 * 4])?,
        })
    }

    fn buffers(&self) -> [&DeviceBuffer; 5] {
        [
            &self.keys,
            &self.ring_keys,
            &self.ring_positions,
            &self.anchor_keys,
            &self.anchor_positions,
        ]
    }

    fn read(&self, session: &Session) -> LabResult<Vec<Vec<u8>>> {
        self.buffers()
            .into_iter()
            .map(|buffer| download(session, buffer, buffer.byte_len()))
            .collect()
    }

    fn restore(&self, session: &Session, image: &[Vec<u8>]) -> LabResult<()> {
        for (buffer, bytes) in self.buffers().into_iter().zip(image) {
            buffer
                .copy_from_bytes(bytes, &session.stream)
                .map_err(|e| e.to_string())?;
        }
        session.stream.synchronize().map_err(|e| e.to_string())
    }
}

/// The key write over one sequence: both ranks' states, the shared inputs
/// and the projection rows' source.
struct KeyWrite<'a> {
    session: &'a Session,
    entry: &'a owned_writes::Entry,
    function: Function,
    table: Vec<i32>,
    ring: usize,
    states: Vec<KeyState>,
    tables: DeviceBuffer,
    weight: DeviceBuffer,
    cos_sin: DeviceBuffer,
}

impl<'a> KeyWrite<'a> {
    fn new(session: &'a Session, entry: &'a owned_writes::Entry) -> LabResult<Self> {
        let module = session.module(&cubin_of(&entry.symbol())?)?;
        let table = table(CONTEXT.div_ceil(PAGE));
        let ring = entry.indexer.ring as usize;
        let states = (0..GROUP)
            .map(|_| KeyState::new(session, &table, ring))
            .collect::<LabResult<Vec<_>>>()?;
        let mut weights = rng(&[0x6b6e]);
        Ok(KeyWrite {
            session,
            entry,
            function: function(&module, &entry.symbol(), entry.plan(1).map_err(refusal)?)?,
            tables: upload(session, &words(table.iter().copied()))?,
            weight: upload(
                session,
                &bf16s((0..INDEX_DIM).map(|_| 0.25 * weights.unit())),
            )?,
            cos_sin: upload(session, &cos_sin(entry))?,
            table,
            ring,
            states,
        })
    }

    /// Writes positions `0..tokens` (projection rows by position) in
    /// launches of `HISTORY_CHUNK` rows on both ranks, without replays.
    fn history(&self, tokens: usize) -> LabResult<()> {
        let positions = (0..tokens).collect::<Vec<_>>();
        let seeds = positions
            .iter()
            .map(|&p| p as u64 | 1 << 40)
            .collect::<Vec<_>>();
        for chunk in positions
            .chunks(HISTORY_CHUNK)
            .zip(seeds.chunks(HISTORY_CHUNK))
        {
            self.launch(chunk.0, chunk.1, None, false)?;
        }
        Ok(())
    }

    /// One launch of rows at `positions` with projection rows of `seeds`
    /// on both ranks, keeping the anchor ring of `anchor` (its end) in the
    /// last spare entry; under the replays when `replay`.
    fn launch(
        &self,
        positions: &[usize],
        seeds: &[u64],
        anchor: Option<usize>,
        replay: bool,
    ) -> LabResult<()> {
        let session = self.session;
        let n = positions.len();
        let first = positions[0];
        let projection = upload(
            session,
            &padded(
                bf16s(seeds.iter().flat_map(|&seed| {
                    let mut rng = rng(&[seed, 0x70]);
                    (0..PROJECTION).map(move |_| rng.unit())
                })),
                n,
            ),
        )?;
        let rope = upload(
            session,
            &padded(words(positions.iter().flat_map(|&p| [p as i32; 3])), n),
        )?;
        let row_batch = upload(session, &padded(words(vec![0; n]), n))?;
        let position_buffer = upload(
            session,
            &padded(words(positions.iter().map(|&p| p as i32)), n),
        )?;
        let live = upload(session, &(n as u32).to_ne_bytes())?;
        let slots = upload(session, &words([0]))?;
        let anchor = anchor.filter(|&end| end > first && end <= first + n);
        let anchor_slots = upload(
            session,
            &words([anchor.map_or(-1, |_| ANCHOR_SLOTS as i32 - 1)]),
        )?;
        let anchor_ends = upload(session, &words([anchor.map_or(-1, |end| end as i32)]))?;
        let plan = self.entry.plan(prepared(n) as u32).map_err(refusal)?;
        let query = self.entry.query();
        for (rank, state) in self.states.iter().enumerate() {
            self.entry
                .check_rank(rank as u32, GROUP as u32)
                .map_err(refusal)?;
            let args = arguments(
                &self.entry.params(),
                vec![
                    ("projection", Value::Buffer(&projection)),
                    ("weight", Value::Buffer(&self.weight)),
                    ("rope_positions", Value::Buffer(&rope)),
                    ("cos_sin", Value::Buffer(&self.cos_sin)),
                    ("row_batch", Value::Buffer(&row_batch)),
                    ("positions", Value::Buffer(&position_buffer)),
                    ("live_rows", Value::Buffer(&live)),
                    ("slot_ids", Value::Buffer(&slots)),
                    ("ring_keys", Value::Buffer(&state.ring_keys)),
                    ("ring_positions", Value::Buffer(&state.ring_positions)),
                    ("anchor_slots", Value::Buffer(&anchor_slots)),
                    ("anchor_ends", Value::Buffer(&anchor_ends)),
                    ("anchor_keys", Value::Buffer(&state.anchor_keys)),
                    ("anchor_positions", Value::Buffer(&state.anchor_positions)),
                    ("block_tables", Value::Buffer(&self.tables)),
                    ("block_keys", Value::Buffer(&state.keys)),
                    ("max_pages", Value::U32(self.table.len() as u32)),
                    ("block_page_stride_bytes", Value::U64(KEY_STRIDE as u64)),
                    ("eps_bits", Value::U32(self.entry.indexer.eps.to_bits())),
                    ("cp_rank", Value::U32(rank as u32)),
                    ("cp_size", Value::U32(GROUP as u32)),
                ],
            )?;
            let kernel = (self.entry.symbol(), &self.function, plan, args);
            if !replay {
                let (symbol, function, plan, args) = &kernel;
                function
                    .launch(&session.stream, launch::config(*plan)?, args)
                    .map_err(|e| e.to_string())?;
                kernel_lab::launched(symbol);
                continue;
            }
            // Every block whose last token is in the launch, once, by its
            // owner; the ring is read and rewritten; the anchored entry's
            // rows of the launch's positions before the anchor end.
            let mut written = Vec::new();
            let first_block = first.saturating_sub(BLOCK - 1).div_ceil(BLOCK);
            for b in (first_block..).take_while(|b| b * BLOCK + BLOCK - 1 < first + n) {
                if let Some((stripe, row)) =
                    owned_block_row(query, rank as u32, b as u64).map_err(refusal)?
                {
                    let at = self.table[stripe as usize] as usize * KEY_STRIDE
                        + row as usize * INDEX_DIM * 2;
                    written.push(at..at + INDEX_DIM * 2);
                }
            }
            let anchored = anchor
                .map(|end| {
                    (end.saturating_sub(self.ring).max(first)..end)
                        .map(|p| (ANCHOR_SLOTS - 1) * self.ring + p % self.ring)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let image = state.read(session)?;
            let replay = Replay::new(session)
                .state(
                    "block_keys",
                    &state.keys,
                    image[0].clone(),
                    WriteMap::of(DType::Bf16).byte_ranges(written)?,
                    Vec::new(),
                )
                .state(
                    "ring_keys",
                    &state.ring_keys,
                    image[1].clone(),
                    WriteMap::of(DType::Bf16).none()?,
                    std::iter::once(0..image[1].len()).collect(),
                )
                .state(
                    "ring_positions",
                    &state.ring_positions,
                    image[2].clone(),
                    WriteMap::of(DType::I32).none()?,
                    std::iter::once(0..image[2].len()).collect(),
                )
                .state(
                    "anchor_keys",
                    &state.anchor_keys,
                    image[3].clone(),
                    WriteMap::of(DType::Bf16).rows(INDEX_DIM, anchored.iter().copied())?,
                    Vec::new(),
                )
                .state(
                    "anchor_positions",
                    &state.anchor_positions,
                    image[4].clone(),
                    WriteMap::of(DType::I32).rows(3, anchored.iter().copied())?,
                    Vec::new(),
                );
            replayed(session, replay, &live, std::slice::from_ref(&kernel))?;
        }
        Ok(())
    }

    fn images(&self) -> LabResult<Vec<Vec<Vec<u8>>>> {
        self.states
            .iter()
            .map(|state| state.read(self.session))
            .collect()
    }

    fn restore(&self, images: &[Vec<Vec<u8>>]) -> LabResult<()> {
        for (state, image) in self.states.iter().zip(images) {
            state.restore(self.session, image)?;
        }
        Ok(())
    }

    /// Runs `case`'s launches after `history`: each rank's block keys,
    /// ring keys and ring positions after them.
    fn run(&self, history: &[Vec<Vec<u8>>], case: &Case) -> LabResult<Units> {
        self.restore(history)?;
        for launch in &case.launches {
            self.launch(
                &case.positions[launch.clone()],
                &case.seeds[launch.clone()],
                None,
                true,
            )?;
        }
        Ok(self
            .images()?
            .into_iter()
            .flat_map(|image| image.into_iter().take(3))
            .collect())
    }
}

/// The units of a stateless stage's case.
fn stateless(session: &Session, stage: &Stage, case: &Case) -> LabResult<Units> {
    match stage {
        Stage::Localize(entry) => localize(session, entry, case),
        Stage::Merge(entry) => merge(session, entry, case),
        Stage::Blocks(entry) => blocks(session, entry, case),
        Stage::Select(entry) => select(session, entry, case),
        Stage::Partials(entry) => partials(session, entry, case),
        Stage::KvWrite(entry) => kv_write(session, entry, case),
        Stage::KeyWrite(_) => Err("the key write keeps state".into()),
    }
}

impl ContextParallel {
    /// The rows of a verify item ending at each of `VERIFY_ENDS`, launched
    /// together (`alone` false) or each alone through the op's one-row
    /// entry.
    fn verify(
        &self,
        sessions: &[Session],
        entry: &Declared,
        seeds: &[u64],
        alone: bool,
    ) -> LabResult<Units> {
        check_geometry(&entry.symbol)?;
        let session = &sessions[0];
        let stage = stage_of(&entry.symbol)?;
        let case = |end| {
            if alone {
                Case::alone(end, seeds)
            } else {
                Case::verify(end, seeds)
            }
        };
        let mut units = Vec::new();
        for end in VERIFY_ENDS {
            units.extend(match &stage {
                Stage::KeyWrite(key) => {
                    let write = KeyWrite::new(session, key)?;
                    write.history(end - seeds.len())?;
                    let history = write.images()?;
                    write.run(&history, &case(end))?
                }
                Stage::Partials(partial) if alone => {
                    let reference = Stage::Partials(one_row_partials(partial)?);
                    stateless(session, &reference, &case(end))?
                }
                stage => stateless(session, stage, &case(end))?,
            });
        }
        Ok(units)
    }
}

impl Adapter for ContextParallel {
    fn kind(&self) -> OpKind {
        OpKind::ContextParallel
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        check_geometry(&entry.symbol)?;
        let session = &sessions[0];
        let case = Case::chunks(rows, ends);
        match stage_of(&entry.symbol)? {
            Stage::KeyWrite(key) => {
                let write = KeyWrite::new(session, &key)?;
                write.history(PREFIX)?;
                let history = write.images()?;
                write.run(&history, &case)
            }
            stage => stateless(session, &stage, &case),
        }
    }

    fn rows(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        self.verify(sessions, entry, seeds, false)
    }

    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        self.verify(sessions, entry, seeds, true)
    }

    /// Key write launches after `PREFIX` cached tokens that keep the anchor
    /// ring in the last spare entry (`anchor_cases`): the whole prompt
    /// anchored at the nearest anchor the claim covers and at its last
    /// granule multiple, and a launch of only the rows up to that nearest
    /// anchor. Every rank's block keys and ring equal those of the launch
    /// without the anchor, and the spare entry holds the ring a prompt
    /// ending at the anchor leaves, history rows before the launch included
    /// (the first spare entry stays untouched, `KeyWrite::launch`'s write
    /// maps).
    fn anchors(&mut self, sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        check_geometry(&entry.symbol)?;
        let session = &sessions[0];
        let Stage::KeyWrite(key) = stage_of(&entry.symbol)? else {
            return Err(format!("{}: keeps no anchors", entry.symbol));
        };
        let write = KeyWrite::new(session, &key)?;
        write.history(PREFIX)?;
        let history = write.images()?;
        let granule = match entry.invariance.chunk {
            Chunk::Granule(granule) => granule as usize,
            Chunk::Any | Chunk::None => 1,
        };
        let span = match entry.invariance.anchors {
            Anchors::SpareAfter { rows } => rows as usize,
            Anchors::Spare => NEAR_ANCHOR,
            other => {
                return Err(format!(
                    "{}: keeps no spare entry ({other:?})",
                    entry.symbol
                ));
            }
        };
        let mut compared = Vec::new();
        let names = ["block keys", "ring keys", "ring positions"];
        for (rows, end) in anchor_cases(span, granule) {
            let case = Case::chunks(rows, &[rows]);
            let plain = write.run(&history, &case)?;
            write.restore(&history)?;
            write.launch(&case.positions, &case.seeds, Some(end), true)?;
            let anchored = write.images()?;
            let ending = Case::chunks((end - PREFIX) as u32, &[(end - PREFIX) as u32]);
            write.run(&history, &ending)?;
            let ended = write.images()?;
            for rank in 0..GROUP {
                for (part, name) in names.iter().enumerate() {
                    compared.push(Compared {
                        label: format!("rank {rank} {name} of {rows} rows with an anchor at {end}"),
                        got: anchored[rank][part].clone(),
                        expected: plain[rank * names.len() + part].clone(),
                    });
                }
                // The spare entry against the ring after a prompt ending at
                // the anchor: keys, then positions.
                for (part, ring) in [(3, 1), (4, 2)] {
                    let spare = &anchored[rank][part];
                    let entry_bytes = spare.len() / ANCHOR_SLOTS;
                    compared.push(Compared {
                        label: format!(
                            "rank {rank} spare {} of {rows} rows against the ring at {end}",
                            names[ring]
                        ),
                        got: spare[(ANCHOR_SLOTS - 1) * entry_bytes..].to_vec(),
                        expected: ended[rank][ring].clone(),
                    });
                }
            }
        }
        Ok(compared)
    }
}

/// The anchor cases (launch rows after `PREFIX`, anchor end) for a spare
/// entry complete past `span` rows and chunk granule `granule`: the whole
/// prompt anchored `span` rows after its start and at its last granule
/// multiple, and a launch of `span` rows anchored at its end.
fn anchor_cases(span: usize, granule: usize) -> Vec<(u32, usize)> {
    let whole = (CONTEXT - PREFIX) as u32;
    let near = PREFIX + span;
    let last = (CONTEXT - 1) / granule * granule;
    let mut cases = vec![(whole, near), (whole, last), (span as u32, near)];
    cases.dedup();
    cases
}

/// The adapter resolves every declared context-parallel entry to a stage,
/// and the fixtures take each group's geometry (no GPU).
#[test]
fn claimed_context_parallel_entries_resolve_to_stages() {
    let groups = groups();
    assert!(!groups.is_empty());
    let claimed = crate::invariance::inventory()
        .into_iter()
        .filter(|d| d.op == OpKind::ContextParallel)
        .collect::<Vec<_>>();
    for declared in &claimed {
        stage_of(&declared.symbol).unwrap();
    }
    for group in &groups {
        for symbol in group.symbols() {
            check_geometry(&symbol).unwrap();
            stage_of(&symbol).unwrap();
            cubin_of(&symbol).unwrap();
        }
        // Each cubin the stages load is a kernel-lab cubin holding the
        // entries.
        for cubin in &group.cubins {
            let entries = kernel_lab::cubin_entries(&cubin.name).unwrap();
            for entry in &cubin.entries {
                assert!(entries.contains(&entry.symbol.as_str()), "{}", entry.symbol);
            }
        }
    }
}

/// The verify partials are the decode partials' instantiation under another
/// symbol: the same body, parameters and threads. So a verify row launched
/// alone through the decode entry is the verify entry's one-row launch.
#[test]
fn verify_partials_share_the_decode_instantiation() {
    for entry in groups().into_iter().flat_map(|group| group.partials) {
        if entry.phase != AttentionPhase::Verify {
            continue;
        }
        let decode = one_row_partials(&entry).unwrap();
        assert_eq!(decode.phase, AttentionPhase::Decode);
        let (a, b) = (entry.entry_spec().unwrap(), decode.entry_spec().unwrap());
        assert_ne!(a.symbol, b.symbol);
        assert_eq!(
            (&a.params, &a.body, a.max_threads, a.min_blocks),
            (&b.params, &b.body, b.max_threads, b.min_blocks)
        );
    }
}

/// The cases' rows, launches and pairs follow their definitions (no GPU):
/// a rows case covers its rows once, the alone case row by row, a chunk
/// case its splits; rank pairs are the rank's pools best first, padded.
#[test]
fn context_parallel_cases_cover_their_rows() {
    let seeds = [7, 8, 9];
    let verify = Case::verify(2052, &seeds);
    assert_eq!(verify.positions, [2049, 2050, 2051]);
    assert_eq!(verify.launches, std::iter::once(0..3).collect::<Vec<_>>());
    let alone = Case::alone(2052, &seeds);
    assert_eq!(alone.launches, [0..1, 1..2, 2..3]);
    assert_eq!([1, 3, 4, 300].map(prepared), [1, 4, 4, 512]);
    assert_eq!(
        anchor_cases(NEAR_ANCHOR, 16),
        [(512, 4100), (512, 4592), (4, 4100)]
    );
    assert_eq!(anchor_cases(16, 4), [(512, 4112), (512, 4604), (16, 4112)]);
    let rows = padded(words([1, 2, 3]), 3);
    assert_eq!(rows.len(), 16);
    assert!(rows[12..].iter().all(|&b| b == POISON));
    let chunks = Case::chunks(512, &[64, 300, 512]);
    assert_eq!(chunks.launches, [0..64, 64..300, 300..512]);
    assert_eq!(chunks.positions[0], PREFIX);
    let parts = vec![vec![vec![1, 2], vec![3, 4]]];
    let two = Case::verify(10, &[1, 2]);
    assert_eq!(units(&two, &parts, 1), [vec![1, 3], vec![2, 4]]);
    for rank in 0..GROUP {
        let pairs = rank_pairs(3, 4096, rank);
        assert_eq!(pairs.len(), TOP * 2);
        let kept = pairs
            .chunks_exact(2)
            .take_while(|p| p[1] != u32::MAX)
            .collect::<Vec<_>>();
        assert_eq!(kept.len(), TOP.min(1024 / 2));
        assert!(
            kept.iter()
                .all(|p| (p[1] as usize / PAGE_POOLS) % GROUP == rank)
        );
        assert!(kept.windows(2).all(|w| {
            let (a, b) = (f32::from_bits(w[0][0]), f32::from_bits(w[1][0]));
            a > b || (a == b && w[0][1] < w[1][1])
        }));
        assert!(
            pairs
                .chunks_exact(2)
                .skip(kept.len())
                .all(|p| p == [f32::NEG_INFINITY.to_bits(), u32::MAX])
        );
    }
}
