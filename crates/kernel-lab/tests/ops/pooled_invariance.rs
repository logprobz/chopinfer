//! The pooled indexer invariance adapter (`OpKind::PooledIndexer`): the
//! lab's pooled indexer entries at the served shape
//! (`pooled_indexer_shapes.rs`: 32 heads of 128, pools of 4, top 2048 with
//! the always-selected tail, 64-token pages, verify items of 6 rows)
//! through the ops SM120 prepares, over the kit's cases
//! (`kit::cases::pooled_indexer`).
//!
//! - Selections and the compaction run as a prepared family
//!   ([`Select`]): a selection's split merge on a backend of the plain
//!   entries; a tree or radix merge on one with the forms, below the rows
//!   that take one split; the one-split form at rows that take one split
//!   (376 on), its rows alone through the one-row launch, a folded form.
//!   Every selection case takes its rows from one source of 512 rows over
//!   shared pages. Ties: a launch over pages that all hold the same pooled
//!   keys (`tied_selection`), where each row picks among equal scores,
//!   against the oracle's lower-index selection, bit for bit.
//! - The prefill key write's rows read the tail slots their item's earlier
//!   rows write, so its cases run here: a chunk case carries the tail state
//!   and pooled keys from chunk to chunk and compares every byte the whole
//!   launch writes; a batch unit is an item, its written bytes with the
//!   other item's key and gate rows changed; the anchor case resumes each
//!   item at a position on a multiple of the group with its tail state
//!   replaced, which must give the whole launch's bytes, while a resume one
//!   position later must not (the case's own negative control).
//! - The verify key write's rows run against decode steps through the
//!   kit's mapping (`key_decoded`) over its served step, 32 items of six
//!   rows; its batch units are two items, as for prefill.
//! - The top-k merges of per-rank selections (at 2 and 4 ranks, as context
//!   parallelism gathers them) run at the served top-k of 2048 per launch
//!   over a case built from the launch's rows (`merge_op_case`: each rank's
//!   pairs the per-rank oracle's selection of the row's pools on its pages,
//!   rows seeing up to 3071 pools); rows (against each row's one-row
//!   launch), chunks and batches as for the selections, and ties over scores
//!   of two values against the merge oracle's lower-pool order.

use super::prepared_invariance::{Family, Prepared};
use crate::invariance::{Adapter, Compared, Units};
use kernel_core::contract::RecurrentPhase;
use kernel_lab::conformance::Images;
use kernel_lab::kit::cases::pooled_indexer as kit;
use kernel_lab::kit::{LabModules, Modules, OpCase};
use kernel_lab::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::invariance::{Declared, OpKind, Scope};
use kernels_sm120::pooled_indexer::{Entry, Kind, merge_radix, merge_tree, rows};
use std::collections::BTreeMap;
use std::ops::Range;

/// Rows of the source every selection case takes rows from.
const SPAN: u32 = 512;
/// Rows of each prefill key write item in a chunk case.
const PROMPT_ROWS: u32 = 192;
/// Rows of each item in the batch and anchor cases of a prefill key write.
const ITEM_ROWS: u32 = 64;

/// Which of the family's entries a symbol is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    /// A selection's split merge, on the plain entries.
    Merge(Kind),
    /// A tree (`true`) or radix merge, on a backend of the plain entries
    /// and that merge, below the rows that take one split.
    Folded(Kind, bool),
    /// The one-split form, at rows that take one split.
    OneSplit(Kind),
    Compact,
    KeyWrite(RecurrentPhase),
    /// The top-k merge of this many ranks' per-rank selections.
    TopK(u32),
}

fn entries() -> Vec<Entry> {
    kernel_lab::families::pooled_indexer::shapes::lab_entries()
}

fn form(entry: &Declared) -> LabResult<Form> {
    let symbol = entry.symbol.as_str();
    for entry in entries() {
        if entry.symbol() == symbol || entry.merge_symbol().as_deref() == Some(symbol) {
            return match entry.kind {
                Kind::Select { .. } => Ok(Form::Merge(entry.kind)),
                Kind::Compact => Ok(Form::Compact),
                Kind::KeyWrite(phase) => Ok(Form::KeyWrite(phase)),
                Kind::TopKMerge { ranks } => Ok(Form::TopK(ranks)),
            };
        }
        let one = std::slice::from_ref(&entry);
        if rows::entries(one, 1).iter().any(|f| f.symbol() == symbol) {
            return Ok(Form::OneSplit(entry.kind));
        }
        if merge_tree::entries(one)
            .iter()
            .any(|f| f.symbol() == symbol)
        {
            return Ok(Form::Folded(entry.kind, true));
        }
        if merge_radix::entries(one)
            .iter()
            .any(|f| f.symbol() == symbol)
        {
            return Ok(Form::Folded(entry.kind, false));
        }
    }
    Err(format!("{symbol}: not a served pooled indexer entry"))
}

/// SM120 over the plain entries: a selection launches its split kernel
/// and split merge.
fn plain() -> Sm120 {
    Sm120::new().with_pooled_indexer(entries())
}

/// SM120 over the plain entries and their tree (`tree`) or radix merges.
fn folded(tree: bool) -> Sm120 {
    let entries = entries();
    if tree {
        plain().with_indexer_merge_tree(merge_tree::entries(&entries))
    } else {
        plain().with_indexer_merge_radix(merge_radix::entries(&entries))
    }
}

/// The selections and the compaction.
pub struct Select;

impl Family for Select {
    fn kind(&self) -> OpKind {
        OpKind::PooledIndexer
    }

    fn case(&self, entry: &Declared, capacity: u32) -> LabResult<OpCase<'static>> {
        match form(entry)? {
            Form::Merge(kind) => kit::select_op_case(&plain(), &entries(), kind, capacity, SPAN),
            Form::Folded(kind, tree) => {
                kit::select_op_case(&folded(tree), &entries(), kind, capacity, SPAN)
            }
            Form::OneSplit(kind) => {
                kit::select_op_case(&kit::sm120(&entries()), &entries(), kind, capacity, SPAN)
            }
            Form::Compact => {
                kit::compact_op_case(&plain(), &entries(), capacity, self.source_rows(entry))
            }
            Form::KeyWrite(_) | Form::TopK(_) => {
                Err(format!("{}: no prepared row case", entry.symbol))
            }
        }
    }

    fn reference(&self, entry: &Declared, capacity: u32) -> LabResult<OpCase<'static>> {
        match form(entry)? {
            // One row splits its pools: a folded form, whose bits every
            // form declares.
            Form::OneSplit(kind) => {
                kit::select_op_case(&kit::sm120(&entries()), &entries(), kind, capacity, SPAN)
            }
            _ => self.case(entry, capacity),
        }
    }

    fn source_rows(&self, entry: &Declared) -> u32 {
        match form(entry) {
            // The compaction's entry holds a decode batch.
            Ok(Form::Compact) => 32,
            _ => SPAN,
        }
    }

    fn row_counts(&self, entry: &Declared, max: u32) -> Vec<u32> {
        match form(entry) {
            Ok(Form::OneSplit(_)) => vec![kit::ONE_SPLIT_CAPACITY, kit::ONE_SPLIT_CAPACITY + 31],
            _ => crate::invariance::row_counts(max),
        }
    }

    fn batch_units(&self, entry: &Declared) -> usize {
        match form(entry) {
            Ok(Form::OneSplit(_)) => kit::ONE_SPLIT_CAPACITY as usize,
            _ => 4,
        }
    }
}

pub struct Pooled(Prepared<Select>);

impl Default for Pooled {
    fn default() -> Self {
        Self(Prepared(Select))
    }
}

/// A key write case of `phase` and its items' row indices: two prefill
/// items of `size` rows, or `size` verify items of their six rows.
fn key_case(phase: RecurrentPhase, size: u32) -> LabResult<(OpCase<'static>, Vec<Range<usize>>)> {
    let (items, item_rows) = match phase {
        RecurrentPhase::Verify { max_rows } => (size, max_rows),
        _ => (2, size),
    };
    let case = kit::key_write_op_case(&plain(), &entries(), phase, items * item_rows)?;
    let n = item_rows as usize;
    let ranges = (0..items as usize)
        .map(|item| item * n..(item + 1) * n)
        .collect();
    Ok((case, ranges))
}

/// The verify key write's served items: its capacity over its rows.
fn verify_items(phase: RecurrentPhase) -> LabResult<u32> {
    let RecurrentPhase::Verify { max_rows } = phase else {
        return Err("not a verify key write".into());
    };
    let entry = entries()
        .into_iter()
        .find(|e| e.kind == Kind::KeyWrite(phase))
        .ok_or("no verify key write entry")?;
    Ok(entry.capacity / max_rows)
}

/// The states a key write leaves, by role path.
const KEY_STATES: [&str; 2] = ["tail.state", "pooled.keys"];

/// The byte ranges of each key write state a launch over `rows` rows of
/// `images` writes or rewrites.
fn key_ranges(
    case: &OpCase<'_>,
    images: &Images,
    rows: u32,
) -> LabResult<Vec<(&'static str, Range<usize>)>> {
    let mut ranges = Vec::new();
    for state in KEY_STATES {
        for range in case.state_ranges(state, images, rows)? {
            ranges.push((state, range));
        }
    }
    Ok(ranges)
}

/// The bytes of `ranges` in `results`, one unit per range.
fn key_units(
    results: &BTreeMap<String, Vec<u8>>,
    ranges: &[(&str, Range<usize>)],
) -> LabResult<Units> {
    ranges
        .iter()
        .map(|(state, range)| {
            results
                .get(*state)
                .and_then(|bytes| bytes.get(range.clone()))
                .map(<[u8]>::to_vec)
                .ok_or_else(|| format!("no bytes {range:?} of {state}"))
        })
        .collect()
}

/// `images` with the key write states of `results`.
fn carried(images: &Images, results: &BTreeMap<String, Vec<u8>>) -> LabResult<Images> {
    let mut images = images.clone();
    for state in KEY_STATES {
        let bytes = results
            .get(state)
            .ok_or_else(|| format!("the launch gave no {state}"))?;
        images = images.with(state, bytes.clone());
    }
    Ok(images)
}

/// Rows `within` of each item, items in order.
fn item_order(items: &[Range<usize>], within: &dyn Fn(usize) -> Range<usize>) -> Vec<usize> {
    items
        .iter()
        .enumerate()
        .flat_map(|(item, rows)| within(item).map(move |n| rows.start + n))
        .collect()
}

impl Pooled {
    /// A prefill key write's items of `rows` rows each, in one launch, or
    /// in chunks ending at `ends` (rows of each item) that carry the tail
    /// state and pooled keys: one unit per byte range the whole launch
    /// writes.
    fn key_chunks(&self, session: &Session, rows: u32, ends: &[u32]) -> LabResult<Units> {
        let functions = LabModules.load(session)?;
        let (case, items) = key_case(RecurrentPhase::Prefill, rows)?;
        let images = case.images();
        let ranges = key_ranges(&case, &images, 2 * rows)?;
        let mut state = images.clone();
        let mut start = 0;
        for &end in ends {
            let order = item_order(&items, &|_| start as usize..end as usize);
            let chunk = case.select(&state, &order)?;
            let results = case.launch(session, &*functions, &chunk, order.len() as u32)?;
            state = carried(&state, &results)?;
            start = end;
        }
        let left = KEY_STATES
            .iter()
            .map(|state_path| Ok((state_path.to_string(), state.bytes(state_path)?.to_vec())))
            .collect::<LabResult<BTreeMap<_, _>>>()?;
        key_units(&left, &ranges)
    }

    /// A key write's two items in one launch, the second's key and gate
    /// rows perturbed by `seeds[1]`: each item's written bytes.
    fn key_batch(
        &self,
        session: &Session,
        phase: RecurrentPhase,
        seeds: &[u64],
    ) -> LabResult<Units> {
        let functions = LabModules.load(session)?;
        // Two items: the probed one beside one whose rows change.
        let size = match phase {
            RecurrentPhase::Verify { .. } => 2,
            _ => ITEM_ROWS,
        };
        let (case, items) = key_case(phase, size)?;
        let images = case.images();
        let roles = case.row_roles(Scope::Entry)?;
        let mut perturbed = images.clone();
        for path in ["key", "gate"] {
            let row = roles
                .inputs
                .iter()
                .find(|(input, _)| input == path)
                .map(|&(_, row)| row)
                .ok_or_else(|| format!("no row input {path}"))?;
            let mut bytes = images.bytes(path)?.to_vec();
            let seed = seeds.get(1).copied().unwrap_or_default();
            // Low mantissa bits of each BF16 element: finite values, other
            // rows for every other seed.
            for (at, half) in bytes[items[1].start * row..items[1].end * row]
                .chunks_exact_mut(2)
                .enumerate()
            {
                let flip = (seed.rotate_left(at as u32 % 64) & 3) as u16 | 1;
                let bits = u16::from_le_bytes([half[0], half[1]]) ^ flip;
                half.copy_from_slice(&bits.to_le_bytes());
            }
            perturbed = perturbed.with(path, bytes);
        }
        let live = items.last().map_or(0, |rows| rows.end) as u32;
        let results = case.launch(session, &*functions, &perturbed, live)?;
        items
            .iter()
            .map(|rows| {
                let alone = case.select(&images, &rows.clone().collect::<Vec<_>>())?;
                let ranges = key_ranges(&case, &alone, rows.len() as u32)?;
                Ok(key_units(&results, &ranges)?.concat())
            })
            .collect()
    }

    /// The prefill key write resumed at anchors `at` (rows of each item)
    /// with the tail state replaced, against the whole launch: the bytes the
    /// whole launch writes, whole then resumed.
    fn resumed(&self, session: &Session, at: [usize; 2]) -> LabResult<(Units, Units)> {
        let functions = LabModules.load(session)?;
        let (case, items) = key_case(RecurrentPhase::Prefill, ITEM_ROWS)?;
        let images = case.images();
        let rows = ITEM_ROWS as usize;
        let ranges = key_ranges(&case, &images, 2 * ITEM_ROWS)?;
        let whole = case.launch(session, &*functions, &images, 2 * ITEM_ROWS)?;
        let prefix = item_order(&items, &|item| 0..at[item]);
        let first = case.launch(
            session,
            &*functions,
            &case.select(&images, &prefix)?,
            prefix.len() as u32,
        )?;
        // A restore from the anchor holds the pooled keys alone: the tail
        // state becomes finite values no row wrote.
        let lost = [0x40_u8, 0x3f].repeat(images.bytes("tail.state")?.len() / 2);
        let state = carried(&images, &first)?.with("tail.state", lost);
        let rest = item_order(&items, &|item| at[item]..rows);
        let resumed = case.launch(
            session,
            &*functions,
            &case.select(&state, &rest)?,
            rest.len() as u32,
        )?;
        Ok((key_units(&whole, &ranges)?, key_units(&resumed, &ranges)?))
    }

    fn key_anchors(&self, session: &Session) -> LabResult<Vec<Compared>> {
        let starts = kit::key_item_starts();
        // The first row of each item whose position is on a multiple of
        // the group, past a third of its rows.
        let aligned = starts.map(|start| {
            (ITEM_ROWS as usize / 3..)
                .find(|n| (start + n) % 4 == 0)
                .expect("an aligned row")
        });
        let (whole, resumed) = self.resumed(session, aligned)?;
        // One position later the item's last group straddles the anchor:
        // a resume without its tail slots must write other bytes, or the
        // case could not tell a lost tail state.
        let (whole_late, resumed_late) = self.resumed(session, aligned.map(|n| n + 1))?;
        if whole_late == resumed_late {
            return Err(format!(
                "resumed at {:?} without tail state, the key write still gives the whole \
                 launch's bytes: the anchor case proves nothing",
                aligned.map(|n| n + 1)
            ));
        }
        Ok(resumed
            .into_iter()
            .zip(whole)
            .enumerate()
            .map(|(i, (got, expected))| Compared {
                label: format!("resumed at rows {aligned:?} range {i}"),
                got,
                expected,
            })
            .collect())
    }

    /// The merge rows of `seeds` in one launch of a case of their rows: each
    /// row's key list.
    fn merge_units(&self, session: &Session, ranks: u32, seeds: &[u64]) -> LabResult<Units> {
        let functions = LabModules.load(session)?;
        let rows = seeds.len() as u32;
        let case = kit::merge_op_case(&plain(), &entries(), ranks, rows, seeds, false)?;
        let results = case.launch(session, &*functions, &case.images(), rows)?;
        let output = results.get("output").ok_or("the merge gave no output")?;
        let row = output.len() / rows as usize;
        Ok(output.chunks_exact(row).map(<[u8]>::to_vec).collect())
    }

    /// Merge rows whose scores take two values, so pools tie within and
    /// across ranks: each row's key list against the merge oracle's, which
    /// keeps the lower pool of equal scores.
    fn merge_ties(&self, session: &Session, ranks: u32) -> LabResult<Vec<Compared>> {
        let functions = LabModules.load(session)?;
        let seeds = (0..u64::from(kit::SPLIT_CAPACITY)).collect::<Vec<_>>();
        let rows = kit::SPLIT_CAPACITY;
        let case = kit::merge_op_case(&plain(), &entries(), ranks, rows, &seeds, true)?;
        let images = case.images();
        let results = case.launch(session, &*functions, &images, rows)?;
        let expected = case.oracle_bits(&images, rows)?;
        let (got, expected) = (
            results.get("output").ok_or("the merge gave no output")?,
            expected.get("output").ok_or("the oracle gave no output")?,
        );
        let row = expected.len() / rows as usize;
        Ok((0..rows as usize)
            .map(|r| Compared {
                label: format!("tied merge row {r} against the lower-pool merge"),
                got: got[r * row..][..row].to_vec(),
                expected: expected[r * row..][..row].to_vec(),
            })
            .collect())
    }

    fn select_ties(&self, session: &Session, entry: &Declared) -> LabResult<Vec<Compared>> {
        let functions = LabModules.load(session)?;
        let rows = match form(entry)? {
            Form::OneSplit(_) => kit::ONE_SPLIT_CAPACITY,
            _ => kit::SPLIT_CAPACITY,
        };
        let case = Select.case(entry, rows)?;
        let images = kit::tied_selection(&case.images())?;
        let results = case.launch(session, &*functions, &images, rows)?;
        let expected = case.oracle_bits(&images, rows)?;
        let (got, expected) = (
            results.get("output").ok_or("the launch gave no output")?,
            expected.get("output").ok_or("the oracle gave no output")?,
        );
        let row = expected.len() / case.capacity() as usize;
        Ok((0..rows as usize)
            .map(|r| Compared {
                label: format!("tied row {r} against the lower-index selection"),
                got: got[r * row..][..row].to_vec(),
                expected: expected[r * row..][..row].to_vec(),
            })
            .collect())
    }
}

impl Adapter for Pooled {
    fn kind(&self) -> OpKind {
        OpKind::PooledIndexer
    }

    fn prompt_rows(&self, entry: &Declared) -> u32 {
        match form(entry) {
            Ok(Form::KeyWrite(_)) => PROMPT_ROWS,
            _ => self.0.prompt_rows(entry),
        }
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        match form(entry)? {
            Form::KeyWrite(RecurrentPhase::Prefill) => self.key_chunks(&sessions[0], rows, ends),
            Form::TopK(ranks) => {
                let mut units = Vec::new();
                let mut start = 0;
                for &end in ends {
                    let seeds = (u64::from(start)..u64::from(end)).collect::<Vec<_>>();
                    units.extend(self.merge_units(&sessions[0], ranks, &seeds)?);
                    start = end;
                }
                Ok(units)
            }
            _ => self.0.chunks(sessions, entry, rows, ends),
        }
    }

    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        match form(entry)? {
            Form::TopK(ranks) => self.merge_units(&sessions[0], ranks, seeds),
            _ => self.0.units(sessions, entry, seeds),
        }
    }

    fn rows(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        match form(entry)? {
            // The selection family has no merge case (its pairs are
            // rank-major blocks): one launch of the rows' merge case.
            Form::TopK(ranks) => self.merge_units(&sessions[0], ranks, seeds),
            Form::KeyWrite(phase @ RecurrentPhase::Verify { .. }) => {
                let functions = LabModules.load(&sessions[0])?;
                let (case, _) = key_case(phase, verify_items(phase)?)?;
                Ok(case
                    .decoded_units(&sessions[0], &*functions, rows_max(entry))?
                    .0)
            }
            _ => self.0.rows(sessions, entry, seeds),
        }
    }

    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        match form(entry)? {
            Form::KeyWrite(phase @ RecurrentPhase::Verify { .. }) => {
                let functions = LabModules.load(&sessions[0])?;
                let (case, _) = key_case(phase, verify_items(phase)?)?;
                Ok(case
                    .decoded_units(&sessions[0], &*functions, rows_max(entry))?
                    .1)
            }
            Form::TopK(ranks) => seeds
                .iter()
                .map(|&seed| {
                    self.merge_units(&sessions[0], ranks, &[seed])?
                        .pop()
                        .ok_or_else(|| format!("{}: no unit", entry.symbol))
                })
                .collect(),
            _ => self.0.alone(sessions, entry, seeds),
        }
    }

    fn row_counts(&self, entry: &Declared, max: u32) -> Vec<u32> {
        match form(entry) {
            // The served verify step: 32 items of six rows.
            Ok(Form::KeyWrite(phase @ RecurrentPhase::Verify { max_rows })) => {
                vec![verify_items(phase).map_or(0, |items| items * max_rows)]
            }
            _ => self.0.row_counts(entry, max),
        }
    }

    fn batch_units(&self, entry: &Declared) -> usize {
        match form(entry) {
            Ok(Form::KeyWrite(_)) => 2,
            _ => self.0.batch_units(entry),
        }
    }

    fn batch(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        match form(entry)? {
            Form::KeyWrite(phase) => self.key_batch(&sessions[0], phase, seeds),
            Form::TopK(ranks) => self.merge_units(&sessions[0], ranks, seeds),
            _ => self.0.batch(sessions, entry, seeds),
        }
    }

    fn ties(&mut self, sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        match form(entry)? {
            Form::Merge(_) | Form::Folded(..) | Form::OneSplit(_) => {
                self.select_ties(&sessions[0], entry)
            }
            Form::TopK(ranks) => self.merge_ties(&sessions[0], ranks),
            _ => Err(format!("{}: no tie case", entry.symbol)),
        }
    }

    fn anchors(&mut self, sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        match form(entry)? {
            Form::KeyWrite(RecurrentPhase::Prefill) => self.key_anchors(&sessions[0]),
            _ => Err(format!("{}: no anchor case", entry.symbol)),
        }
    }
}

/// An entry's one-row maximum.
fn rows_max(entry: &Declared) -> u32 {
    match entry.invariance.rows {
        kernels_sm120::invariance::Rows::OneRow { max } => max,
        kernels_sm120::invariance::Rows::None => 0,
    }
}

/// Every declared entry that claims something resolves to its form; each
/// selection and compaction case launches its entry and claims its
/// declaration; the key write cases prepare, take their chunk orders and
/// map verify rows to decode steps (no GPU).
#[test]
fn pooled_cases_launch_their_entries() {
    let declared = kernel_lab::declared::pooled_indexer();
    let mut forms = Vec::new();
    for entry in declared
        .iter()
        .filter(|e| !e.invariance.claims().is_empty())
    {
        let form = form(entry).unwrap();
        forms.push(form);
        let launches = |case: &OpCase<'_>| {
            assert!(
                case.info().symbols.iter().any(|(s, _)| *s == entry.symbol),
                "{form:?} {}: {:?}",
                entry.symbol,
                case.info().symbols
            );
            assert_eq!(case.info().guarantees, entry.invariance, "{}", entry.symbol);
        };
        match form {
            Form::KeyWrite(phase) => {
                let size = match phase {
                    RecurrentPhase::Verify { .. } => verify_items(phase).unwrap(),
                    _ => ITEM_ROWS,
                };
                let (case, items) = key_case(phase, size).unwrap();
                launches(&case);
                let images = case.images();
                let order = item_order(&items, &|_| 0..3);
                case.select(&images, &order).unwrap();
                assert!(
                    !key_ranges(&case, &images, order.len() as u32)
                        .unwrap()
                        .is_empty()
                );
            }
            Form::TopK(ranks) => {
                let case = kit::merge_op_case(&plain(), &entries(), ranks, 3, &[5, 9, 200], false)
                    .unwrap();
                launches(&case);
                // The oracle merges every rank's pairs of each row.
                assert_eq!(case.oracle_bits(&case.images(), 3).unwrap().len(), 1);
                // The rows case launches merge cases of its rows, never the
                // selection family's, at each row count and alone.
                assert!(Select.case(entry, 2).is_err());
                let adapter = Pooled::default();
                for count in Adapter::row_counts(&adapter, entry, rows_max(entry))
                    .into_iter()
                    .chain([1])
                {
                    let seeds = (0..u64::from(count)).collect::<Vec<_>>();
                    let case =
                        kit::merge_op_case(&plain(), &entries(), ranks, count, &seeds, false)
                            .unwrap();
                    launches(&case);
                    case.oracle_bits(&case.images(), count).unwrap();
                }
            }
            _ => {
                let rows = Select.row_counts(entry, 64)[0];
                launches(&Select.case(entry, rows).unwrap());
                let source = Select.case(entry, Select.source_rows(entry)).unwrap();
                let reference = Select.reference(entry, 1).unwrap();
                reference.select(&source.images(), &[7]).unwrap();
            }
        }
    }
    // The selections in every form, the compaction and both key writes
    // that claim something.
    for wanted in [
        Form::Compact,
        Form::KeyWrite(RecurrentPhase::Prefill),
        Form::KeyWrite(RecurrentPhase::Verify { max_rows: 6 }),
        Form::TopK(2),
        Form::TopK(4),
    ] {
        assert!(forms.contains(&wanted), "{wanted:?}: {forms:?}");
    }
    let global = Kind::Select { ranks: None };
    for wanted in [
        Form::Merge(global),
        Form::Folded(global, false),
        Form::Folded(global, true),
        Form::OneSplit(global),
    ] {
        assert!(forms.contains(&wanted), "{wanted:?}: {forms:?}");
    }
}
