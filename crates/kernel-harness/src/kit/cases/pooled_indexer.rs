//! Cases of the pooled indexer paths SM120 serves through new forms
//! (`kernel_core::pooled_indexer`): the key write at context-parallel group
//! rank 0 of 1 and at a rank of a group of two, and the weighted selection
//! through its forms (the radix and tree merges of the split lists, and
//! the one-split kernel), with the SM120 selection of the lab's pooled
//! indexer entries and the forms the lab builds for them ([`sm120`]).
//!
//! Key write: two items whose prefill rows complete a group from tail
//! slots, cross a page and leave a new tail, over pooled key pages and tail
//! states that hold a pattern and are ID inputs (page tables, slot ids).
//! Raw keys are drawn so every normalized value rounds to one BF16 value
//! within the norm's arithmetic allowance ([`History`]): a kernel within
//! the norm's gate stores the oracle's BF16 keys, so the tail state compares
//! as bits, and each pooled key within the pooling oracle's bound. A group
//! rank writes the pooled keys of its own pages only.
//!
//! Selection: two items over shuffled pages of integer pooled keys, whose
//! F32 scores the host reproduces in the kernel's arithmetic (the dots are
//! exact), so key lists and per-rank pairs compare as bits against the
//! oracle's selection of those scores. The capacity sets the form: 16 rows
//! split each row's pools (the radix or tree merge folds them), 376 rows
//! take one split (the one-split kernel). Each case is declared only when
//! the prepared op launches the form it covers.

use super::super::decode::{Compared, DecodeLaunch, Verified};
use super::super::{Decoded, OpCase, Required, Spec, bf16_bytes, bf16_values};
use crate::LabResult;
use crate::conformance::{Expected, Holes, Images, Rows, Write};
use crate::poison::WriteMap;
use kernel_core::attention::BlockTables;
use kernel_core::backend::{Backend, Need, Prepared, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16, f64_to_bf16};
use kernel_core::context_parallel::ContextParallelQuery;
use kernel_core::contract::{DType, RecurrentPhase, StaticQuery};
use kernel_core::oracle::OracleOutput;
use kernel_core::pooled_indexer::{
    IndexerKeyWriteBuffers, IndexerMergeBuffers, IndexerSelectBuffers, IndexerSelectionMode,
    IndexerTailState, PooledIndexerQuery, PooledKeyPages, SelectionCompactBuffers,
};
use kernel_core::pooled_indexer_oracle::{self as oracle, KeyWriteOperands, TailState};
use kernel_core::properties::Property;
use kernels_sm120::backend::Sm120;
use kernels_sm120::pooled_indexer::{Entry, Kind, merge_radix, merge_tree, rows};
use std::ops::Range;

/// The key write cases' rows.
pub const KEY_CAPACITY: u32 = 16;
/// Selection rows that split each row's pools.
pub const SPLIT_CAPACITY: u32 = 16;
/// The fewest selection rows that take one split.
pub const ONE_SPLIT_CAPACITY: u32 = 376;

const HEADS: usize = 32;
const DIM: usize = 128;
const GROUP: usize = 4;
const PAGE: usize = 64;
/// Pools of one page.
const PAGE_POOLS: usize = PAGE / GROUP;
/// Pairs of a per-rank row (`max_top_k / group`).
const MAX_KEPT: usize = 512;
/// The selections' runtime top-k: each row keeps four pools. The merges
/// run at the served top-k ([`merge_top_k`]).
const TOP_K: u32 = 16;
/// A page of pooled keys and 512 decoy bytes.
const STRIDE: usize = PAGE_POOLS * DIM * 2 + 512;
/// The items of every case.
const ITEMS: usize = 2;
/// Each selection item's first position: in the fourth page, so every
/// group rank of a group of up to four owns pools each row sees.
const SELECT_FIRST: usize = 250;
/// Each key write item's first position: the first completes a group of
/// the second page from three tail slots; the second completes one of the
/// second page from two slots and crosses into the third.
const KEY_STARTS: [usize; ITEMS] = [67, 126];
/// The key write cases' live counts: from two rows each launch completes
/// a group on a page of the second group rank and leaves a tail.
const KEY_LIVE: [u32; 5] = [KEY_CAPACITY, 0, 2, KEY_CAPACITY / 2, KEY_CAPACITY - 1];
/// Each key write item's state entry, of [`STATE_ENTRIES`].
const KEY_SLOTS: [i32; ITEMS] = [3, 1];
const STATE_ENTRIES: usize = 4;
/// The key and gate row strides, wider than a row.
const KEY_STRIDE: usize = DIM + 32;
const GATE_STRIDE: usize = DIM + 8;

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

/// The entry of `kind` of `entries`, which must have the geometry the
/// cases are written for: 32 heads of 128, pools of 4, 64-token pages and
/// 512 kept pairs per row.
fn entry_of(entries: &[Entry], kind: Kind) -> LabResult<&Entry> {
    let entry = entries
        .iter()
        .find(|entry| entry.kind == kind)
        .ok_or_else(|| format!("no pooled indexer entry {kind:?}"))?;
    let g = entry.geometry;
    let expected = [HEADS, DIM, GROUP, PAGE, MAX_KEPT].map(|n| n as u32);
    if [g.heads, g.head_dim, g.group, g.page_size, g.max_kept()] != expected {
        return Err(format!(
            "{}: the pooled indexer cases need a geometry of {expected:?} (heads, head dim, \
             group, page, kept pairs)",
            entry.name
        ));
    }
    Ok(entry)
}

/// SM120 over the pooled indexer `entries` with every form it builds for
/// them: the one-row one-split forms, the tree merges and the global
/// selection's radix merge.
pub fn sm120(entries: &[Entry]) -> Sm120 {
    Sm120::new()
        .with_pooled_indexer(entries.to_vec())
        .with_indexer_one_split(rows::entries(entries, 1))
        .with_indexer_merge_tree(merge_tree::entries(entries))
        .with_indexer_merge_radix(merge_radix::entries(entries))
}

/// The query of the entry of `kind` of `entries`.
fn query(entries: &[Entry], kind: Kind) -> LabResult<PooledIndexerQuery> {
    let entry = entry_of(entries, kind)?;
    match entry.query() {
        StaticQuery::PooledIndexer(query) => Ok(query),
        _ => Err(format!("{kind:?} is not a pooled indexer query")),
    }
}

/// The prepared op's launches include `symbol`, the form its case covers.
fn launches<B>(prepared: &Prepared<B>, symbol: &str) -> LabResult<()> {
    if prepared
        .info()
        .symbols
        .iter()
        .any(|(launched, _)| launched == symbol)
    {
        Ok(())
    } else {
        Err(format!("the op does not launch {symbol}"))
    }
}

/// The required ops: the prefill key write at rank 0 of 1 and at rank 1
/// of 2, the global selection through its radix merge and one-split form,
/// and the per-rank selections through their tree merges (rank 1 of 2 and
/// 3 of 4) and one-split form (rank 1 of 2).
pub fn required<'a>(entries: &[Entry]) -> LabResult<Vec<Required<'a>>> {
    let entry = |kind| entry_of(entries, kind).cloned();
    let mut ops = Vec::new();
    let write = query(entries, Kind::KeyWrite(RecurrentPhase::Prefill))?;
    for (rank, size) in [(0, 1), (1, 2)] {
        let need = Need::rows(KEY_CAPACITY);
        ops.push(
            Required::new(
                format!("indexer key write prefill rank {rank} of {size} rows {KEY_CAPACITY}"),
                need,
                move |backend, need| backend.indexer_key_write(&write, rank, size, need),
                move |prepared| key_write_case(write, rank, size, need.capacity, prepared),
            )
            .required(),
        );
    }
    // The verify key write at rank 0 of 1 over its served items (32 of six
    // rows), its rows against decode steps.
    let max_rows = entry(Kind::KeyWrite(RecurrentPhase::Prefill))?
        .geometry
        .verify_rows;
    let verify = query(entries, Kind::KeyWrite(RecurrentPhase::Verify { max_rows }))?;
    let decode = query(entries, Kind::KeyWrite(RecurrentPhase::Decode))?;
    let rows = verify_capacity(verify)?;
    let items = rows / max_rows;
    ops.push(
        Required::new(
            format!("indexer key write verify rank 0 of 1 rows {rows}"),
            Need::rows(rows),
            move |backend, need| backend.indexer_key_write(&verify, 0, 1, need),
            move |prepared| {
                Ok(key_write_case(verify, 0, 1, rows, prepared)?
                    .item_rows(ITEM_ROWS)
                    .without_properties(&[Property::Batch], ITEM_ROWS))
            },
        )
        .with_decode(move |backend| {
            backend
                .indexer_key_write(&decode, 0, 1, Need::rows(items))
                .map(|prepared| key_decoded(decode, max_rows as usize, items as usize, prepared))
        })
        .required(),
    );
    // The top-k merges of per-rank selections at 2 and 4 ranks, at the
    // served top-k.
    for ranks in [2, 4] {
        let merge = merge_query(entries, ranks);
        let per_rank = query(entries, Kind::Select { ranks: Some(ranks) })?;
        ops.push(
            Required::new(
                format!("indexer top-k merge of {ranks} ranks rows {SPLIT_CAPACITY}"),
                Need::rows(SPLIT_CAPACITY),
                {
                    let merge = merge.clone();
                    move |backend, need| {
                        let op = merge.clone().map_err(kernel_core::backend::Refused::Rule)?;
                        let top_k = merge_top_k(op).map_err(kernel_core::backend::Refused::Rule)?;
                        backend.indexer_merge(&op, top_k, need)
                    }
                },
                move |prepared| {
                    let seeds = (0..u64::from(SPLIT_CAPACITY)).collect::<Vec<_>>();
                    merge_case(
                        merge.clone()?,
                        per_rank,
                        ranks,
                        SPLIT_CAPACITY,
                        &seeds,
                        false,
                        prepared,
                    )
                },
            )
            .required(),
        );
    }
    for (ranks, rank, capacity) in [
        (None, 0, SPLIT_CAPACITY),
        (None, 0, ONE_SPLIT_CAPACITY),
        (Some(2), 1, SPLIT_CAPACITY),
        (Some(4), 3, SPLIT_CAPACITY),
        (Some(2), 1, ONE_SPLIT_CAPACITY),
    ] {
        let select = entry(Kind::Select { ranks })?;
        let op = query(entries, Kind::Select { ranks })?;
        let (form, symbol) = match (select.splits(capacity), ranks) {
            (1, _) => ("one-split", rows::Entry::of(&select, 1).map(|e| e.symbol())),
            (_, None) => (
                "radix merge",
                merge_radix::Entry::of(&select).map(|e| e.symbol()),
            ),
            (_, Some(_)) => (
                "tree merge",
                merge_tree::Entry::of(&select).map(|e| e.symbol()),
            ),
        };
        let symbol = symbol.map_err(|e| e.0)?;
        let mode = match ranks {
            None => "global".to_owned(),
            Some(ranks) => format!("rank {rank} of {ranks}"),
        };
        ops.push(
            Required::new(
                format!("indexer selection {mode} rows {capacity} {form}"),
                Need::rows(capacity),
                move |backend, need| backend.indexer_select(&op, TOP_K, rank, need),
                move |prepared| {
                    launches(&prepared, &symbol)?;
                    select_case(op, rank, capacity, capacity, prepared)
                },
            )
            .required(),
        );
    }
    Ok(ops)
}

/// The case of the selection of the entry `kind` of `entries` on `backend`
/// at `capacity` rows, its images those of a case of `span` rows
/// ([`select_case`]): a per-rank selection at its last group rank. For the
/// invariance adapters, which launch rows of one source through cases of
/// any capacity (the capacity sets the form SM120 runs).
pub fn select_op_case(
    backend: &Sm120,
    entries: &[Entry],
    kind: Kind,
    capacity: u32,
    span: u32,
) -> LabResult<OpCase<'static>> {
    let Kind::Select { ranks } = kind else {
        return Err(format!("{kind:?} is not a selection"));
    };
    let op = query(entries, kind)?;
    let rank = ranks.map_or(0, |ranks| ranks - 1);
    let prepared = backend
        .indexer_select(&op, TOP_K, rank, Need::rows(capacity))
        .map_err(|e| e.to_string())?;
    select_case(op, rank, capacity, span, prepared)
}

/// A selection case's `images` with every page holding the first page's
/// pooled keys: pool `r` of each page then scores as pool `r` of every
/// other, so each row's top-k chooses among equal scores and only the
/// lower-index rule orders them. The oracle selects over the same images.
pub fn tied_selection(images: &Images) -> LabResult<Images> {
    let keys = images.bytes("pooled.keys")?;
    let first = keys
        .get(..STRIDE)
        .ok_or("the pooled keys hold no page")?
        .to_vec();
    let tied = keys
        .chunks(STRIDE)
        .flat_map(|page| first[..page.len()].to_vec())
        .collect::<Vec<_>>();
    Ok(images.with("pooled.keys", tied))
}

/// The top-k merge query of the entry of `entries` merging `ranks`
/// per-rank selections.
fn merge_query(entries: &[Entry], ranks: u32) -> LabResult<ContextParallelQuery> {
    let entry = entry_of(entries, Kind::TopKMerge { ranks })?;
    match entry.query() {
        StaticQuery::ContextParallel(query) => Ok(query),
        _ => Err(format!(
            "the merge of {ranks} ranks is not a context-parallel query"
        )),
    }
}

/// A merge row of `seed` at runtime top-k `top_k`: its position, drawn
/// from `[top_k / 2, 6 * top_k)` (from [`SELECT_FIRST`] over at least 256
/// positions), and the F32 scores of its visible pools, quarter steps in
/// [0, 6), or only 0 and 1/4 when `tied`, so equal scores across pools and
/// ranks are common. At the served top-k of 2048 (512 pools kept) a row sees
/// 256 to 3071 pools: some keep every pool, some lose pools in the merge
/// only, and past 4096 (two ranks) or 8192 positions (four) each rank's own
/// selection drops pools as well.
fn merge_row(seed: u64, tied: bool, top_k: u32) -> (u64, Vec<f32>) {
    let first = SELECT_FIRST.max(top_k as usize / 2);
    let span = (6 * top_k as usize).saturating_sub(first).max(256);
    let mut draw = Draw(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let position = (first + (draw.next() % span as u64) as usize) as u64;
    let visible = (position as usize + 1) / GROUP;
    let steps = if tied { 2 } else { 24 };
    let scores = (0..visible)
        .map(|_| (draw.next() % steps) as f32 * 0.25)
        .collect();
    (position, scores)
}

/// The runtime top-k the merges serve: the geometry's maximum, the served
/// index top-k of 2048.
fn merge_top_k(op: ContextParallelQuery) -> LabResult<u32> {
    match op {
        ContextParallelQuery::TopKMerge { max_top_k, .. } => Ok(max_top_k),
        _ => Err("not a top-k merge".into()),
    }
}

/// The case of the top-k merge of `ranks` per-rank selections on `backend`
/// over `capacity` rows at the served top-k ([`merge_top_k`]), row `r` the
/// merge row of `seeds[r % len]` ([`merge_row`]): each rank's pairs are the per-rank oracle's selection
/// of the pools on its pages (page `p` on rank `p % ranks`), gathered
/// rank-major at `capacity` rows a rank, and the expected key list is the
/// merge oracle's, bit for bit. The pairs are rank-major blocks, not rows of
/// a row op, so the case reports the row properties not exercised; the ops
/// invariance adapter builds a case per launch from its rows' seeds.
pub fn merge_op_case(
    backend: &Sm120,
    entries: &[Entry],
    ranks: u32,
    capacity: u32,
    seeds: &[u64],
    tied: bool,
) -> LabResult<OpCase<'static>> {
    let op = merge_query(entries, ranks)?;
    let per_rank = query(entries, Kind::Select { ranks: Some(ranks) })?;
    let prepared = backend
        .indexer_merge(&op, merge_top_k(op)?, Need::rows(capacity))
        .map_err(|e| e.to_string())?;
    merge_case(op, per_rank, ranks, capacity, seeds, tied, prepared)
}

/// [`merge_op_case`] over the prepared merge `op` of `ranks` selections
/// `per_rank`.
fn merge_case<'a>(
    op: ContextParallelQuery,
    per_rank: PooledIndexerQuery,
    ranks: u32,
    capacity: u32,
    seeds: &[u64],
    tied: bool,
    prepared: Prepared<IndexerMergeBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let top_k = merge_top_k(op)?;
    let width = op.list_width().ok_or("a key list merge has a width")? as usize;
    if seeds.is_empty() {
        return Err("a merge case needs a row seed".into());
    }
    let rows = capacity as usize;
    let size = ranks as usize;
    let mut pairs = vec![0_u8; size * rows * MAX_KEPT * 8];
    let mut positions = Vec::with_capacity(rows);
    for row in 0..rows {
        let (position, scores) = merge_row(seeds[row % seeds.len()], tied, top_k);
        positions.push(position as i32);
        for rank in 0..size {
            let owned = (0..scores.len())
                .map(|pool| pool / PAGE_POOLS % size == rank)
                .collect::<Vec<_>>();
            let kept = oracle::select_per_rank(per_rank, top_k, position, &scores, &owned)
                .map_err(|e| e.0)?;
            let at = (rank * rows + row) * MAX_KEPT * 8;
            for (i, (score, pool)) in kept.into_iter().enumerate() {
                pairs[at + i * 8..][..4].copy_from_slice(&score.to_bits().to_le_bytes());
                pairs[at + i * 8 + 4..][..4].copy_from_slice(&pool.to_le_bytes());
            }
        }
    }
    let roles = IndexerMergeBuffers {
        pairs: Spec::input(DType::I32, (Rows::Fixed(size * rows), 2 * MAX_KEPT), pairs),
        rank_rows: capacity,
        positions: Some(Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(&positions),
        )),
        live_rows: Spec::live(),
        // -1 entries are valid values.
        output: Spec::output(DType::I32, (Rows::Live, width)).filled_with(&[0x5a]),
    };
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let pairs = images.bytes("pairs")?;
        let positions = images.i32s("positions")?;
        let mut out = vec![0_u8; rows * width * 4];
        for row in 0..live as usize {
            let lists = (0..size)
                .map(|rank| {
                    let at = (rank * rows + row) * MAX_KEPT * 8;
                    pairs[at..at + MAX_KEPT * 8]
                        .chunks_exact(8)
                        .map(|pair| {
                            (
                                f32::from_bits(u32::from_le_bytes(pair[..4].try_into().unwrap())),
                                i32::from_le_bytes(pair[4..].try_into().unwrap()),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let refs = lists.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let position = u64::try_from(positions[row]).map_err(|e| e.to_string())?;
            let list = oracle::merge(op, top_k, position, &refs).map_err(|e| e.0)?;
            out[row * width * 4..][..list.len() * 4].copy_from_slice(&i32_bytes(&list));
        }
        Ok(vec![("output".to_owned(), Expected::Bits(out))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?
        .oracle(oracle)
        .without_row_properties(MERGE_ROWS))
}

/// Why a merge case runs no row property check.
const MERGE_ROWS: &str = "a merge's pairs are rank-major blocks of every rank's rows, which the \
                          kit's row checks cannot move; the ops invariance adapter builds a case \
                          per launch";

/// Each key write item's first position.
pub fn key_item_starts() -> [usize; ITEMS] {
    KEY_STARTS
}

/// The case of the key write of `phase` on `backend` at group rank 0 of 1
/// over `capacity` rows ([`key_write_case`]); a verify case maps its rows
/// to decode steps on the same backend ([`key_decoded`]).
pub fn key_write_op_case(
    backend: &Sm120,
    entries: &[Entry],
    phase: RecurrentPhase,
    capacity: u32,
) -> LabResult<OpCase<'static>> {
    let op = query(entries, Kind::KeyWrite(phase))?;
    let prepared = backend
        .indexer_key_write(&op, 0, 1, Need::rows(capacity))
        .map_err(|e| e.to_string())?;
    let case = key_write_case(op, 0, 1, capacity, prepared)?;
    let RecurrentPhase::Verify { max_rows } = phase else {
        return Ok(case);
    };
    let decode = query(entries, Kind::KeyWrite(RecurrentPhase::Decode))?;
    let items = capacity / max_rows;
    let prepared = backend
        .indexer_key_write(&decode, 0, 1, Need::rows(items))
        .map_err(|e| e.to_string())?;
    Ok(case.item_rows(ITEM_ROWS).decoded(key_decoded(
        decode,
        max_rows as usize,
        items as usize,
        prepared,
    )))
}

/// The rows the verify key write `op` serves: its items of `max_rows`.
pub fn verify_capacity(op: PooledIndexerQuery) -> LabResult<u32> {
    match op {
        PooledIndexerQuery::KeyWrite {
            phase: RecurrentPhase::Verify { .. },
            capacity,
            ..
        } => Ok(capacity),
        _ => Err("not a verify key write".into()),
    }
}

/// The case of the selection compaction on `backend` over `capacity`
/// output rows from key lists of `span` rows: each list a pattern of
/// positions and holes, each output row's source a list row.
pub fn compact_op_case(
    backend: &Sm120,
    entries: &[Entry],
    capacity: u32,
    span: u32,
) -> LabResult<OpCase<'static>> {
    let op = query(entries, Kind::Compact)?;
    let PooledIndexerQuery::CompactSelection { width, .. } = op else {
        return Err("the compaction entry compacts selections".into());
    };
    let prepared = backend
        .indexer_compact(&op, Need::rows(capacity))
        .map_err(|e| e.to_string())?;
    let (rows, width, span) = (
        capacity as usize,
        width as usize,
        span.max(capacity) as usize,
    );
    let lists = (0..span * width)
        .map(|at| {
            if at % 9 == 4 {
                -1
            } else {
                ((at * 31 + at / width * 7) % 65_536) as i32
            }
        })
        .collect::<Vec<_>>();
    let sources = (0..rows)
        .map(|row| ((row * 7 + 3) % span) as i32)
        .collect::<Vec<_>>();
    let roles = SelectionCompactBuffers {
        lists: Spec::input(DType::I32, (Rows::Fixed(span), width), i32_bytes(&lists)),
        source: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&sources)),
        live_rows: Spec::live(),
        output: Spec::output(DType::I32, (Rows::Live, width)).filled_with(&[0x5a]),
    };
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let lists = images.bytes("lists")?;
        let sources = images.i32s("source")?;
        let row = width * 4;
        let mut out = vec![0_u8; rows * row];
        for (at, &source) in sources.iter().enumerate().take(live as usize) {
            let source = usize::try_from(source).map_err(|e| e.to_string())?;
            out[at * row..][..row].copy_from_slice(
                lists
                    .get(source * row..(source + 1) * row)
                    .ok_or_else(|| format!("source row {source} lies past the lists"))?,
            );
        }
        Ok(vec![("output".to_owned(), Expected::Bits(out))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// Why a key write case's rows are not rows of a row op: an item's rows
/// read the tail slots its earlier rows write, so moving or splitting a
/// row moves its item's state. The ops invariance adapter
/// (`tests/ops/pooled_invariance.rs`) carries the state across chunks and
/// checks items beside other items.
const ITEM_ROWS: &str = "a key write row reads the tail slots its item's earlier rows write; the \
                         ops invariance adapter carries them";

/// How a verify key write's rows map to one-row decode key writes over
/// `prepared`, the decode op of query `decode` (`kit::decode`):
///
/// - The verify step runs every item's `max_rows` rows; a claim of fewer
///   rows than an item holds is refused, as the step cannot cut an item.
/// - Its row `t` of item `b` is a decode step of that row from the state
///   decode would hold before it: for row 0 the item's slot in the
///   committed state the verify step read, after that its checkpoint of
///   row `t - 1` in the state it left, which the decode step advances in
///   place (the item's slot id names that entry).
/// - The decode step's live tail slots (those before the row's position
///   in its group, and its own) equal the verify checkpoint's of row `t`,
///   and the pooled key of a group the row completes equals the verify
///   step's. Rows that equal a decode step from the state before them
///   equal the chain of decode steps from the slot. Slots past the live
///   ones follow the history and are not claimed.
fn key_decoded<'a>(
    decode: PooledIndexerQuery,
    max_rows: usize,
    items: usize,
    prepared: Prepared<IndexerKeyWriteBuffers<View>>,
) -> LabResult<Decoded<'a>> {
    let case = key_write_case(decode, 0, 1, items as u32, prepared)?;
    let cut = move |images: &Images, max: u32| -> LabResult<Images> {
        if (max as usize) < max_rows {
            return Err(format!(
                "a verify item of {max_rows} rows takes no claim of {max} rows"
            ));
        }
        Ok(images.clone())
    };
    let launches = move |verified: &Verified<'_>, base: &Images| -> LabResult<Vec<DecodeLaunch>> {
        let images = verified.images;
        let slots = images.i32s("tail.slot_ids")?;
        let checkpoints = images.i32s("tail.checkpoint_ids")?;
        let tables = images.i32s("pooled.page_tables.tables")?;
        let stripes = tables.len() / slots.len();
        // Row 0 starts from the committed state the verify step read, never
        // from what it left.
        let committed = images.bytes("tail.state")?.to_vec();
        let left = verified
            .results
            .get("tail.state")
            .ok_or("the verify step gave no tail state")?;
        let id = |id: i32| usize::try_from(id).map_err(|_| "a negative state entry".to_owned());
        // `path`'s base image with `bytes` at byte `at`.
        let placed = |path: &str, at: usize, bytes: &[u8]| -> LabResult<Vec<u8>> {
            let mut image = base.bytes(path)?.to_vec();
            image[at..at + bytes.len()].copy_from_slice(bytes);
            Ok(image)
        };
        let mut launches = Vec::new();
        for item in 0..slots.len() {
            let (rows, start) = live_rows(images, verified.live, item)?;
            for (t, row) in rows.enumerate() {
                let position = start as usize + t;
                let from = id(if t == 0 {
                    slots[item]
                } else {
                    checkpoints[item * max_rows + t - 1]
                })?;
                let checkpoint = id(checkpoints[item * max_rows + t])?;
                let state = if t == 0 { &committed } else { left };
                let mut images_of = base
                    .with("tail.state", state.clone())
                    .with(
                        "tail.slot_ids",
                        placed("tail.slot_ids", item * 4, &i32_bytes(&[from as i32]))?,
                    )
                    .with(
                        "row_batch",
                        placed("row_batch", 0, &i32_bytes(&[item as i32]))?,
                    );
                for (path, bytes) in [
                    ("key", KEY_STRIDE * 2),
                    ("gate", GATE_STRIDE * 2),
                    ("positions", 4),
                ] {
                    images_of = images_of.with(
                        path,
                        placed(path, 0, &images.bytes(path)?[row * bytes..][..bytes])?,
                    );
                }
                let mut compared = Vec::new();
                for slot in 0..(position + 1) % GROUP {
                    for (verify, decode) in slot_rows(checkpoint, slot)
                        .into_iter()
                        .zip(slot_rows(from, slot))
                    {
                        compared.push(Compared {
                            verify: ("tail.state".into(), verify),
                            decode: ("tail.state".into(), decode),
                        });
                    }
                }
                if position % GROUP == GROUP - 1 {
                    let page = id(tables[item * stripes + position / PAGE])?;
                    let at = page * STRIDE + position / GROUP % PAGE_POOLS * DIM * 2;
                    compared.push(Compared {
                        verify: ("pooled.keys".into(), at..at + DIM * 2),
                        decode: ("pooled.keys".into(), at..at + DIM * 2),
                    });
                }
                launches.push(DecodeLaunch {
                    label: format!("item {item} row {t} at position {position}"),
                    images: images_of,
                    compared,
                });
            }
        }
        Ok(launches)
    };
    Ok(Decoded::new(case, cut, launches))
}

/// Small integers as BF16.
fn integers(values: impl Iterator<Item = i32>) -> Vec<u16> {
    values.map(|value| f32_to_bf16(value as f32)).collect()
}

/// `pages` pages of pooled keys: pool row `r` of page `p` holds code `c =
/// (7 p + 3) % 32 * 16 + r` as `[c + 1, c * 5 % 7 + 1, (c * 3 + d) % 5
/// ...]` in BF16, then decoy bytes, so the dots with [`queries`] are small
/// exact integers and the pages of up to 32 differ out of page order: a
/// read of other pages ranks other pools first.
fn pooled_keys(pages: usize) -> Vec<u8> {
    let mut image = vec![0x7f; pages * STRIDE];
    for page in 0..pages {
        for row in 0..PAGE_POOLS {
            let code = ((7 * page + 3) % 32 * PAGE_POOLS + row) as i32;
            let key = integers((0..DIM as i32).map(|d| match d {
                0 => code + 1,
                1 => code * 5 % 7 + 1,
                _ => (code * 3 + d) % 5,
            }));
            let at = page * STRIDE + row * DIM * 2;
            image[at..at + DIM * 2].copy_from_slice(&bf16_bytes(&key));
        }
    }
    image
}

/// `rows` rows' queries: even heads `[1, h / 2 % 3]`, odd heads `[-1, 1]`,
/// the rest of each head 0, so the dots are small integers of both signs.
fn queries(rows: usize) -> Vec<u16> {
    integers((0..rows * HEADS * DIM).map(|at| {
        let (head, d) = (at / DIM % HEADS, at % DIM);
        match (head % 2, d) {
            (0, 0) => 1,
            (0, 1) => (head / 2 % 3) as i32,
            (1, 0) => -1,
            (1, 1) => 1,
            _ => 0,
        }
    }))
}

/// Each item's local page per stripe of `size` pages, `stripes` per item,
/// distinct and in reverse across the first `items * stripes` pages.
fn local_tables(items: usize, stripes: usize) -> Vec<i32> {
    let used = items * stripes;
    (0..used).map(|at| (used - 1 - at) as i32).collect()
}

/// A selection's operands, decoded once per oracle call.
struct Operands {
    q: Vec<f32>,
    weights: Vec<f32>,
    keys: Vec<u8>,
}

impl Operands {
    fn of(images: &Images) -> LabResult<Operands> {
        let decode = |bits: Vec<u16>| bits.into_iter().map(bf16_to_f32).collect();
        Ok(Operands {
            q: decode(images.u16s("q")?),
            weights: decode(images.u16s("head_weights")?),
            keys: images.bytes("pooled.keys")?.to_vec(),
        })
    }

    /// The F32 score for row `row` of the pool at byte `at` of the pooled
    /// keys: `sum_h w_h relu(dot_h * f32(128^-0.5))` with `w_h = bf16 *
    /// f32(32^-0.5)`, F32 FMA from zero in head order; the dots are exact.
    fn score(&self, row: usize, at: usize) -> LabResult<f32> {
        let scale = (1.0 / (DIM as f64).sqrt()) as f32;
        let head_scale = (1.0 / (HEADS as f64).sqrt()) as f32;
        let key = self
            .keys
            .get(at..at + DIM * 2)
            .ok_or_else(|| format!("pooled key byte {at} lies past the pool"))?
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect::<Vec<_>>();
        let mut score = 0_f32;
        for head in 0..HEADS {
            let w = self.weights[row * HEADS + head] * head_scale;
            let query = &self.q[(row * HEADS + head) * DIM..][..DIM];
            let dot: f64 = query
                .iter()
                .zip(&key)
                .map(|(&q, &k)| f64::from(q) * f64::from(k))
                .sum();
            let x = dot as f32 * scale;
            score = w.mul_add(if x > 0.0 { x } else { 0.0 }, score);
        }
        Ok(score)
    }
}

/// The case of a selection `op` at group rank `rank` over `capacity` rows
/// of two items at consecutive positions from [`SELECT_FIRST`], laid out
/// as a case of `span` rows lays them (its pages and tables): a case of
/// fewer rows takes rows of a larger one's images.
fn select_case<'a>(
    op: PooledIndexerQuery,
    rank: u32,
    capacity: u32,
    span: u32,
    prepared: Prepared<IndexerSelectBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let PooledIndexerQuery::Select { mode, .. } = op else {
        return Err("a selection case selects by a weighted selection query".into());
    };
    let size = match mode {
        IndexerSelectionMode::Global => 1,
        IndexerSelectionMode::PerRank { ranks } => ranks as usize,
    };
    let rows = capacity as usize;
    let item_rows = (span.max(capacity) as usize / ITEMS).max(1);
    let last = SELECT_FIRST + item_rows - 1;
    let stripes = (last / PAGE + 1).div_ceil(size);
    let pages = 2 * ITEMS * stripes + 1;
    let items = (0..rows)
        .map(|row| (row / item_rows).min(ITEMS - 1) as i32)
        .collect::<Vec<_>>();
    let positions = (0..rows)
        .map(|row| (SELECT_FIRST + row % item_rows) as i32)
        .collect::<Vec<_>>();
    let weights = integers((0..rows * HEADS).map(|at| (at / HEADS + at % HEADS) as i32 % 3 + 1));
    let (output, row_elements) = match mode {
        IndexerSelectionMode::Global => (
            DType::I32,
            op.list_width().ok_or("a selection has a key list")? as usize,
        ),
        // U32 (score bits, pool) words.
        IndexerSelectionMode::PerRank { .. } => (DType::I32, 2 * MAX_KEPT),
    };
    let workspace = prepared.info().bytes_of("workspace").unwrap_or(0) as usize;
    let roles = IndexerSelectBuffers {
        q: Spec::input(
            DType::Bf16,
            (Rows::Live, HEADS * DIM),
            bf16_bytes(&queries(rows)),
        ),
        head_weights: Some(Spec::input(
            DType::Bf16,
            (Rows::Live, HEADS),
            bf16_bytes(&weights),
        )),
        pooled: PooledKeyPages {
            keys: Spec::input(DType::U8, (Rows::Fixed(pages), STRIDE), pooled_keys(pages)),
            pages: pages as u32,
            page_stride: STRIDE as u64,
            page_tables: BlockTables {
                tables: Spec::ids(
                    (Rows::Fixed(ITEMS), stripes),
                    local_tables(ITEMS, stripes),
                    &[("pooled.keys", STRIDE)],
                    Holes::Never,
                ),
                sequences: ITEMS as u32,
                max_pages: stripes as u32,
            },
        },
        row_batch: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&items)),
        positions: Spec::input(DType::I32, (Rows::Live, 1), i32_bytes(&positions)),
        live_rows: Spec::live(),
        workspace: if workspace > 0 {
            Spec::scratch(workspace)
        } else {
            Spec::untouched(16)
        },
        // -1 entries and pools are valid values.
        output: Spec::output(output, (Rows::Live, row_elements)).filled_with(&[0x5a]),
    };
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let tables = images.i32s("pooled.page_tables.tables")?;
        let items = images.i32s("row_batch")?;
        let positions = images.i32s("positions")?;
        let operands = Operands::of(images)?;
        let mut out = vec![0_u8; rows * row_elements * 4];
        for row in 0..live as usize {
            let item = usize::try_from(items[row]).map_err(|e| e.to_string())?;
            let position = u64::try_from(positions[row]).map_err(|e| e.to_string())?;
            let visible = (position as usize + 1) / GROUP;
            let mut scores = vec![0_f32; visible];
            let mut owned = vec![false; visible];
            for pool in 0..visible {
                let physical = pool / PAGE_POOLS;
                if physical % size != rank as usize {
                    continue;
                }
                let page = usize::try_from(tables[item * stripes + physical / size])
                    .map_err(|e| e.to_string())?;
                let at = page * STRIDE + pool % PAGE_POOLS * DIM * 2;
                scores[pool] = operands.score(row, at)?;
                owned[pool] = true;
            }
            let bytes = match mode {
                IndexerSelectionMode::Global => {
                    i32_bytes(&oracle::select(op, TOP_K, position, &scores).map_err(|e| e.0)?)
                }
                IndexerSelectionMode::PerRank { .. } => {
                    oracle::select_per_rank(op, TOP_K, position, &scores, &owned)
                        .map_err(|e| e.0)?
                        .into_iter()
                        .flat_map(|(score, pool)| {
                            [score.to_bits().to_le_bytes(), pool.to_le_bytes()]
                        })
                        .flatten()
                        .collect()
                }
            };
            out[row * row_elements * 4..][..bytes.len()].copy_from_slice(&bytes);
        }
        Ok(vec![("output".to_owned(), Expected::Bits(out))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// A deterministic draw of the key write's operands.
struct Draw(u64);

impl Draw {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn pick<T: Copy>(&mut self, values: &[T]) -> T {
        values[(self.next() % values.len() as u64) as usize]
    }
}

/// One item's history up to its last row: raw keys and gate logits per
/// position, and the BF16 normalized keys. A raw row is `m + s * k_d` with
/// an integer shift m, a scale s of 1/2, 1 or 2 and signs k_d, half +1 and
/// half -1: its mean is m and its variance s^2 exactly, so with the weights
/// 1/2 or 1 and the biases -2, 1.5, 2.5 or 3 of [`norm_operands`] its
/// normalized values lie within 2e-6 of BF16 values of magnitude at least
/// 1/2. Every row is checked to round to one BF16 value within the norm's
/// arithmetic allowance `2^-22 K S`.
struct History {
    raw: Vec<u16>,
    gates: Vec<u16>,
    keys: Vec<u16>,
}

impl History {
    fn draw(
        draw: &mut Draw,
        query: PooledIndexerQuery,
        positions: usize,
        weight: &[f64],
        bias: &[f64],
    ) -> LabResult<History> {
        let (mut raw, mut keys) = (Vec::new(), Vec::new());
        for position in 0..positions {
            let shift = draw.pick(&[-3.0_f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
            let scale = draw.pick(&[0.5_f32, 1.0, 2.0]);
            let mut signs = [[1.0_f32; DIM / 2], [-1.0; DIM / 2]].concat();
            for i in (1..DIM).rev() {
                signs.swap(i, (draw.next() % (i as u64 + 1)) as usize);
            }
            let row = signs
                .iter()
                .map(|k| f32_to_bf16(shift + scale * k))
                .collect::<Vec<_>>();
            let norm = oracle::key_norm(query, &widen(&row), weight, bias).map_err(|e| e.0)?;
            for (&value, &s) in norm.values.iter().zip(&norm.sum_abs_products) {
                let allowance = 2_f64.powi(-22) * DIM as f64 * s;
                let low = f64_to_bf16(value - allowance);
                if low != f64_to_bf16(value + allowance) {
                    return Err(format!("position {position}: an ambiguous normalized key"));
                }
                keys.push(low);
            }
            raw.extend(row);
        }
        Ok(History {
            raw,
            gates: (0..positions * DIM)
                .map(|_| f32_to_bf16(2.0 * draw.unit()))
                .collect(),
            keys,
        })
    }
}

/// LayerNorm weight and bias whose sums and differences are BF16 values of
/// magnitude at least 1/2 ([`History`]).
fn norm_operands(draw: &mut Draw) -> (Vec<u16>, Vec<u16>) {
    let weight = (0..DIM)
        .map(|_| f32_to_bf16(draw.pick(&[0.5, 1.0])))
        .collect();
    let bias = (0..DIM)
        .map(|_| f32_to_bf16(draw.pick(&[-2.0, 1.5, 2.5, 3.0])))
        .collect();
    (weight, bias)
}

/// The rows of `item` among the first `live`, and the first one's
/// position.
fn live_rows(images: &Images, live: u32, item: usize) -> LabResult<(Range<usize>, u64)> {
    let items = images.i32s("row_batch")?;
    let positions = images.i32s("positions")?;
    let rows = (0..live as usize)
        .filter(|&row| items[row] == item as i32)
        .collect::<Vec<_>>();
    let (Some(&first), Some(&last)) = (rows.first(), rows.last()) else {
        return Ok((0..0, 0));
    };
    let start = u64::try_from(positions[first]).map_err(|e| e.to_string())?;
    Ok((first..last + 1, start))
}

/// What a key write launch at `live` rows writes and rewrites, for group
/// rank `rank` of `size` over tables of `stripes` local pages per item:
/// the pooled rows of the groups it completes on the rank's pages and the
/// tail slots it writes without reading (`written`), and the tail slots
/// it reads and rewrites in place (`updated`).
struct KeyMaps {
    pools: Vec<(usize, u64, Range<usize>)>,
    tail: Vec<Range<usize>>,
    updated: Vec<Range<usize>>,
}

const STATE_BYTES: usize = 2 * (GROUP - 1) * DIM * 2;

/// The byte ranges of tail slot `slot`'s key and gate rows in entry
/// `entry`.
fn slot_rows(entry: usize, slot: usize) -> [Range<usize>; 2] {
    let key = entry * STATE_BYTES + slot * DIM * 2;
    let gate = key + (GROUP - 1) * DIM * 2;
    [key..key + DIM * 2, gate..gate + DIM * 2]
}

/// A key write case's pages and phase: group rank, group size, local
/// pages per item, and a verify step's rows per item (its checkpoints).
type KeyGeometry = (usize, usize, usize, Option<usize>);

fn key_maps(
    images: &Images,
    live: u32,
    (rank, size, stripes, checkpoints): KeyGeometry,
) -> LabResult<KeyMaps> {
    let tables = images.i32s("pooled.page_tables.tables")?;
    let slots = images.i32s("tail.slot_ids")?;
    let mut maps = KeyMaps {
        pools: Vec::new(),
        tail: Vec::new(),
        updated: Vec::new(),
    };
    for item in 0..slots.len() {
        let (rows, start) = live_rows(images, live, item)?;
        if rows.is_empty() {
            continue;
        }
        let end = start as usize + rows.len();
        for position in (start as usize..end).filter(|p| p % GROUP == GROUP - 1) {
            let pool = position / GROUP;
            let physical = position / PAGE;
            if physical % size != rank {
                continue;
            }
            let page = usize::try_from(tables[item * stripes + physical / size])
                .map_err(|e| e.to_string())?;
            let at = page * STRIDE + pool % PAGE_POOLS * DIM * 2;
            maps.pools.push((item, pool as u64, at..at + DIM * 2));
        }
        // A verify step leaves the slot and writes each row's live tail
        // (the slots of its group up to its own, `key_write.cuh`) to its
        // checkpoint entry; a row that completes a group writes none, and
        // slots past the live ones keep the entry's bytes.
        if let Some(max_rows) = checkpoints {
            let ids = images.i32s("tail.checkpoint_ids")?;
            for row in 0..rows.len() {
                let entry =
                    usize::try_from(ids[item * max_rows + row]).map_err(|e| e.to_string())?;
                let tail = (start as usize + row + 1) % GROUP;
                for slot in 0..tail {
                    maps.tail.extend(slot_rows(entry, slot));
                }
            }
            continue;
        }
        // Slots before the launch that a completed straddling group reads.
        let offset = start as usize % GROUP;
        let read = if offset > 0 && rows.len() >= GROUP - offset {
            offset
        } else {
            0
        };
        let entry = usize::try_from(slots[item]).map_err(|e| e.to_string())?;
        let tail = end % GROUP;
        let base = end - tail;
        for slot in (0..tail).filter(|slot| base + slot >= start as usize) {
            let ranges = slot_rows(entry, slot);
            if slot < read {
                maps.updated.extend(ranges);
            } else {
                maps.tail.extend(ranges);
            }
        }
    }
    Ok(maps)
}

/// Positions from each item's start that a decode or verify case's rows
/// take: a verify item's rows, which decode steps repeat one at a time.
const STEP_POSITIONS: usize = 8;
/// Tail state entries of a decode or verify case of `items` items: each
/// item's verify checkpoints from the bottom, then a spare and a slot per
/// item ([`step_slot`]). The run's high-ID rerun moves every slot but the
/// last up, the first item's (whose decode row writes a tail slot, exact
/// bits) among them, and leaves the checkpoints (their ids are a plain
/// input it does not move).
fn step_entries(items: usize) -> usize {
    items * STEP_POSITIONS + 2 * items
}

/// Item `item`'s slot among [`step_entries`]: after the checkpoints, a
/// spare then the slot for each item, so the last item's slot is the top
/// entry and every other slot moves in the rerun.
fn step_slot(items: usize, item: usize) -> i32 {
    (items * STEP_POSITIONS + 2 * item + 1) as i32
}

/// Item `item`'s first position: [`KEY_STARTS`] for the first two, then
/// each start one further on, so items begin at every offset in a group.
fn item_start(item: usize) -> usize {
    KEY_STARTS[item % ITEMS] + item / ITEMS
}

/// The case of a key write `op` at group rank `rank` of `size` over
/// `capacity` rows of items from [`item_start`]: in prefill two items of
/// `capacity / 2` rows; in decode `capacity` items of one row each (one
/// past the start), in verify `capacity / max_rows` items of `max_rows`
/// rows, from histories and tail state entries every decode and verify case
/// of as many items shares, so a decode case takes a verify case's rows and
/// states. A verify step writes each row's state to its checkpoint entry.
fn key_write_case<'a>(
    op: PooledIndexerQuery,
    rank: u32,
    size: u32,
    capacity: u32,
    prepared: Prepared<IndexerKeyWriteBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let PooledIndexerQuery::KeyWrite { pooling, phase, .. } = op else {
        return Err("a key write case writes by a key write query".into());
    };
    if (pooling.group as usize, pooling.width as usize) != (GROUP, DIM) {
        return Err("the key write case is drawn for the lab's pools".into());
    }
    let (rank, size) = (rank as usize, size as usize);
    let rows = capacity as usize;
    let (items, item_rows, checkpoints) = match phase {
        RecurrentPhase::Prefill => (ITEMS, rows / ITEMS, None),
        RecurrentPhase::Decode => (rows, 1, None),
        RecurrentPhase::Verify { max_rows } => (
            rows / max_rows as usize,
            max_rows as usize,
            Some(max_rows as usize),
        ),
    };
    if items == 0 || rows != items * item_rows && phase != RecurrentPhase::Prefill {
        return Err(format!(
            "a decode or verify key write case holds whole items, not {rows} rows"
        ));
    }
    if item_rows > STEP_POSITIONS && phase != RecurrentPhase::Prefill {
        return Err(format!(
            "a verify item of {item_rows} rows passes its history"
        ));
    }
    let starts = (0..items).map(item_start).collect::<Vec<_>>();
    let last = starts.iter().copied().max().unwrap_or(0);
    let (end, entries) = match phase {
        RecurrentPhase::Prefill => (last + item_rows, STATE_ENTRIES),
        _ => (last + STEP_POSITIONS, step_entries(items)),
    };
    // Each item's first row: a decode case's one row past the start, so
    // its first item writes a tail slot (the start completes a group).
    let lead = usize::from(phase == RecurrentPhase::Decode);
    let slots = match phase {
        RecurrentPhase::Prefill => KEY_SLOTS.to_vec(),
        _ => (0..items).map(|item| step_slot(items, item)).collect(),
    };
    let firsts = starts.iter().map(|start| start + lead).collect::<Vec<_>>();
    let stripes = (end.div_ceil(PAGE)).div_ceil(size);
    let pages = 2 * items * stripes + 1;
    let mut draw = Draw(0x9e37_79b9_7f4a_7c15);
    let (weight, bias) = norm_operands(&mut draw);
    let position = (0..GROUP * DIM)
        .map(|_| f32_to_bf16(draw.unit()))
        .collect::<Vec<_>>();
    let histories = (0..items)
        .map(|_| History::draw(&mut draw, op, end, &widen(&weight), &widen(&bias)))
        .collect::<LabResult<Vec<_>>>()?;
    // Rows: each item's rows in order, at stride, the padding a pattern.
    let strided = |stride: usize, row: &dyn Fn(&History) -> &[u16]| {
        let mut out = bf16_values(41, rows * stride);
        for at in 0..rows {
            let (item, n) = (at / item_rows, at % item_rows);
            let p = firsts[item] + n;
            out[at * stride..][..DIM].copy_from_slice(&row(&histories[item])[p * DIM..][..DIM]);
        }
        bf16_bytes(&out)
    };
    let key = strided(KEY_STRIDE, &|history| &history.raw);
    let gate = strided(GATE_STRIDE, &|history| &history.gates);
    // Tail states: a pattern, then each item's live slots before its first
    // row.
    let mut state = bf16_values(42, entries * STATE_BYTES / 2);
    for item in 0..items {
        let start = firsts[item];
        let entry = slots[item] as usize * STATE_BYTES / 2;
        for slot in 0..start % GROUP {
            let p = start - start % GROUP + slot;
            let (key, gate) = pooling
                .tail_slot_offsets(slot as u32)
                .ok_or("a tail slot lies inside the banks")?;
            let history = &histories[item];
            state[entry + key as usize..][..DIM].copy_from_slice(&history.keys[p * DIM..][..DIM]);
            state[entry + gate as usize..][..DIM].copy_from_slice(&history.gates[p * DIM..][..DIM]);
        }
    }
    let pool_image = bf16_bytes(&bf16_values(43, pages * STRIDE / 2));
    let geometry = (rank, size, stripes, checkpoints);
    let tail_write = move || {
        Write::Map(Box::new(move |images: &Images, live| {
            WriteMap::of(DType::Bf16).byte_ranges(key_maps(images, live, geometry)?.tail)
        }))
    };
    let state = match checkpoints {
        Some(_) => Spec::state(
            DType::Bf16,
            (Rows::Fixed(entries), STATE_BYTES / 2),
            bf16_bytes(&state),
            tail_write(),
        ),
        None => Spec::state_updating(
            DType::Bf16,
            (Rows::Fixed(entries), STATE_BYTES / 2),
            bf16_bytes(&state),
            tail_write(),
            move |images, live| Ok(key_maps(images, live, geometry)?.updated),
        ),
    };
    // Item b's checkpoint of row r: entry `b * STEP_POSITIONS + r`. A plain
    // input: the slot ids are the state's ID input.
    let checkpoint_ids = checkpoints.map(|max_rows| {
        Spec::input(
            DType::I32,
            (Rows::Fixed(items), max_rows),
            i32_bytes(
                &(0..items * max_rows)
                    .map(|at| (at / max_rows * STEP_POSITIONS + at % max_rows) as i32)
                    .collect::<Vec<_>>(),
            ),
        )
    });
    let roles = IndexerKeyWriteBuffers {
        key: Spec::input(DType::Bf16, (Rows::Live, KEY_STRIDE), key),
        key_stride: KEY_STRIDE as u32,
        gate: Spec::input(DType::Bf16, (Rows::Live, GATE_STRIDE), gate),
        gate_stride: GATE_STRIDE as u32,
        norm_weight: Spec::input(DType::Bf16, (Rows::Fixed(1), DIM), bf16_bytes(&weight)),
        norm_bias: Spec::input(DType::Bf16, (Rows::Fixed(1), DIM), bf16_bytes(&bias)),
        position: Spec::input(
            DType::Bf16,
            (Rows::Fixed(GROUP), DIM),
            bf16_bytes(&position),
        ),
        row_batch: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(
                &(0..rows)
                    .map(|at| (at / item_rows) as i32)
                    .collect::<Vec<_>>(),
            ),
        ),
        positions: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(
                &(0..rows)
                    .map(|at| (firsts[at / item_rows] + at % item_rows) as i32)
                    .collect::<Vec<_>>(),
            ),
        ),
        live_rows: Spec::live(),
        tail: IndexerTailState {
            state,
            entries: entries as u32,
            slot_ids: Spec::ids(
                (Rows::Fixed(items), 1),
                slots.clone(),
                &[("tail.state", STATE_BYTES)],
                Holes::Never,
            ),
            checkpoint_ids,
        },
        pooled: PooledKeyPages {
            keys: Spec::state(
                DType::Bf16,
                (Rows::Fixed(pages), STRIDE / 2),
                pool_image,
                Write::Map(Box::new(move |images, live| {
                    let maps = key_maps(images, live, geometry)?;
                    WriteMap::of(DType::Bf16)
                        .byte_ranges(maps.pools.into_iter().map(|(_, _, at)| at))
                })),
            ),
            pages: pages as u32,
            page_stride: STRIDE as u64,
            // Pooled keys compare within tolerance, which cannot tell a
            // moved-page rerun from a stale read, so the tables move no
            // pool; the write maps place each pooled row by them.
            page_tables: BlockTables {
                tables: Spec::ids(
                    (Rows::Fixed(items), stripes),
                    local_tables(items, stripes),
                    &[],
                    Holes::Never,
                ),
                sequences: items as u32,
                max_pages: stripes as u32,
            },
        },
    };
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let maps = key_maps(images, live, geometry)?;
        let mut state = images.bytes("tail.state")?.to_vec();
        let slots = images.i32s("tail.slot_ids")?;
        let pool = images.u16s("pooled.keys")?;
        let mut pooled = OracleOutput {
            values: widen(&pool),
            sum_abs_products: vec![0.0; pool.len()],
            reduction_lengths: vec![1; pool.len()],
        };
        let (weight, bias) = (
            widen(&images.u16s("norm_weight")?),
            widen(&images.u16s("norm_bias")?),
        );
        let position = widen(&images.u16s("position")?);
        let (key, gate) = (images.u16s("key")?, images.u16s("gate")?);
        let states = images.u16s("tail.state")?;
        for (item, &slot) in slots.iter().enumerate() {
            let (rows, start) = live_rows(images, live, item)?;
            if rows.is_empty() {
                continue;
            }
            let entry = usize::try_from(slot).map_err(|e| e.to_string())?;
            let held = &states[entry * STATE_BYTES / 2..][..STATE_BYTES / 2];
            let bank = (GROUP - 1) * DIM;
            let tail = TailState {
                keys: widen(&held[..bank]),
                gates: widen(&held[bank..]),
            };
            let rows_of = |source: &[u16], stride: usize| {
                rows.clone()
                    .flat_map(|row| widen(&source[row * stride..][..DIM]))
                    .collect::<Vec<_>>()
            };
            let written = oracle::key_write(
                op,
                start,
                &tail,
                KeyWriteOperands {
                    raw: &rows_of(&key, KEY_STRIDE),
                    gates: &rows_of(&gate, GATE_STRIDE),
                    norm_weight: &weight,
                    norm_bias: &bias,
                    position: &position,
                },
            )
            .map_err(|e| e.0)?;
            // The tail after a row, bit for bit: its keys are the oracle's
            // single BF16 rounding, its gates the inputs.
            let bits = |after: &TailState| {
                after
                    .keys
                    .iter()
                    .chain(&after.gates)
                    .flat_map(|&value| f64_to_bf16(value).to_le_bytes())
                    .collect::<Vec<_>>()
            };
            if let Some(max_rows) = checkpoints {
                // A verify step writes each row's live tail slots.
                let ids = images.i32s("tail.checkpoint_ids")?;
                for (row, after) in written.checkpoints.iter().enumerate() {
                    let entry =
                        usize::try_from(ids[item * max_rows + row]).map_err(|e| e.to_string())?;
                    let after = bits(after);
                    for slot in 0..(start as usize + row + 1) % GROUP {
                        for range in slot_rows(entry, slot) {
                            let at = range.start % STATE_BYTES;
                            state[range.clone()].copy_from_slice(&after[at..at + range.len()]);
                        }
                    }
                }
            } else {
                let after = bits(
                    written
                        .checkpoints
                        .last()
                        .ok_or("a key write of live rows keeps a tail")?,
                );
                for range in maps.tail.iter().chain(&maps.updated) {
                    if range.start / STATE_BYTES == entry {
                        let at = range.start % STATE_BYTES;
                        state[range.clone()].copy_from_slice(&after[at..at + range.len()]);
                    }
                }
            }
            for (owner, number, at) in &maps.pools {
                if *owner != item {
                    continue;
                }
                let reference = written
                    .pools
                    .iter()
                    .find(|written| written.pool == *number)
                    .ok_or_else(|| format!("the oracle completes no pool {number}"))?;
                let first = at.start / 2;
                for d in 0..DIM {
                    pooled.values[first + d] = reference.key.values[d];
                    pooled.sum_abs_products[first + d] = reference.key.sum_abs_products[d];
                    pooled.reduction_lengths[first + d] = reference.key.reduction_lengths[d];
                }
            }
        }
        Ok(vec![
            ("tail.state".to_owned(), Expected::Bits(state)),
            ("pooled.keys".to_owned(), Expected::Values(pooled)),
        ])
    };
    let case = OpCase::new(prepared, capacity, roles)?.oracle(oracle);
    Ok(match phase {
        RecurrentPhase::Prefill => case
            .live_counts(KEY_LIVE.to_vec())
            .without_properties(&[Property::Chunk, Property::Batch], ITEM_ROWS),
        RecurrentPhase::Decode => case.no_write(
            "pooled.keys",
            1,
            "the first item's one row writes a tail slot, inside its group",
        ),
        RecurrentPhase::Verify { .. } => case.no_write(
            "tail.state",
            1,
            "the first item's first row completes its group, leaving no live tail slot to \
             checkpoint",
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Entries of the geometry the cases are written for, under a test
    /// profile: the served replicated-indexer shape (top 2048 with the
    /// always-selected tail, BF16 position embedding, verify items of 6
    /// rows).
    fn entries() -> Vec<Entry> {
        let profile = kernels_sm120::Profile::synthetic(kernels_sm120::ProfileNames {
            name: "pooled-indexer-cases",
            tag: "pidx_cases",
        });
        kernels_sm120::pooled_indexer::entries(
            profile,
            kernels_sm120::pooled_indexer::Geometry {
                heads: HEADS as u32,
                head_dim: DIM as u32,
                group: GROUP as u32,
                page_size: PAGE as u32,
                max_top_k: 2048,
                always_tail: true,
                position_dtype: DType::Bf16,
                key_norm_eps: 1e-6,
                verify_rows: 6,
            },
        )
    }

    /// The key write cases complete groups on both ranks of a pair, so a
    /// write that ignores its group rank stores pools the other rank owns;
    /// at every positive live count the second rank completes a group and
    /// a tail slot is written.
    #[test]
    fn key_write_items_complete_groups_on_both_ranks_of_a_pair() {
        let item_rows = KEY_CAPACITY as usize / ITEMS;
        let rows = (0..KEY_CAPACITY as usize)
            .map(|at| KEY_STARTS[at / item_rows] + at % item_rows)
            .collect::<Vec<_>>();
        let completes = |p: &usize| p % GROUP == GROUP - 1;
        let owners = rows
            .iter()
            .filter(|p| completes(p))
            .map(|p| p / PAGE % 2)
            .collect::<Vec<_>>();
        assert!(owners.contains(&0) && owners.contains(&1), "{owners:?}");
        for live in KEY_LIVE.into_iter().filter(|&live| live > 0) {
            let live = &rows[..live as usize];
            assert!(live.iter().any(|p| completes(p) && p / PAGE % 2 == 1));
            assert!(live.iter().any(|p| !completes(p)));
        }
        // Each item straddles a group it completes from tail slots.
        for start in KEY_STARTS {
            assert!(start % GROUP > 0);
            assert!(item_rows >= GROUP - start % GROUP);
        }
    }

    /// The served verify step's tail writes are each row's live slots in
    /// its checkpoint entry, as `key_write.cuh` stores them: a row that
    /// completes a group (item 0's first, at position 67) writes none.
    #[test]
    fn verify_checkpoints_map_their_live_tail_slots() {
        let phase = RecurrentPhase::Verify { max_rows: 6 };
        let entries = entries();
        let capacity = verify_capacity(query(&entries, Kind::KeyWrite(phase)).unwrap()).unwrap();
        let case = key_write_op_case(&sm120(&entries), &entries, phase, capacity).unwrap();
        let images = case.images();
        let ids = images.i32s("tail.checkpoint_ids").unwrap();
        let mut expected = Vec::new();
        for item in 0..capacity as usize / 6 {
            for row in 0..6 {
                let entry = ids[item * 6 + row] as usize;
                for slot in 0..(item_start(item) + row + 1) % GROUP {
                    expected.extend(slot_rows(entry, slot));
                }
            }
        }
        // Adjacent ranges merge, as the write map holds them.
        let merged = |mut ranges: Vec<Range<usize>>| {
            ranges.sort_by_key(|r| r.start);
            let mut out: Vec<Range<usize>> = Vec::new();
            for r in ranges {
                match out.last_mut() {
                    Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
                    _ => out.push(r),
                }
            }
            out
        };
        let got = merged(case.state_ranges("tail.state", &images, capacity).unwrap());
        assert_eq!(got, merged(expected));
        assert_eq!(item_start(0) % GROUP, GROUP - 1);
        let first = ids[0] as usize * STATE_BYTES..(ids[0] as usize + 1) * STATE_BYTES;
        assert!(
            got.iter()
                .all(|r| r.end <= first.start || r.start >= first.end)
        );
    }

    /// The capacities take the forms the cases name: rows that split each
    /// row's pools, and the fewest rows that take one split.
    #[test]
    fn capacities_take_their_forms() {
        let entries = entries();
        let select = entry_of(&entries, Kind::Select { ranks: None }).unwrap();
        assert!(select.splits(SPLIT_CAPACITY) > 1);
        assert_eq!(select.splits(ONE_SPLIT_CAPACITY), 1);
        assert!(select.splits(ONE_SPLIT_CAPACITY - 1) > 1);
        // Entries of another geometry are refused.
        let mut other = entries.clone();
        other[0].geometry.heads = 16;
        assert!(entry_of(&other, other[0].kind).is_err());
        // Every selection row sees more pools than it keeps, some on the
        // pages of every rank of a group of four.
        const {
            assert!((SELECT_FIRST + 1) / GROUP > (TOP_K as usize / GROUP) * 4);
            assert!(SELECT_FIRST >= 3 * PAGE);
        }
    }
}
