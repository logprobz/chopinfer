//! Invariance tests generated from the catalogs' declarations
//! (`kernels_sm120::invariance`). One harness per property iterates every
//! declared entry that claims it and hands the entry to the adapter of its
//! op kind, which builds random rows (with their auxiliary inputs) and
//! launches them; the harness picks the cases from the declaration and
//! compares bitwise.
//!
//! - Chunk: a prompt prefilled in one chunk against the same prompt split
//!   at the declared granule (any row for `Chunk::Any`), row by row.
//! - Rows: an R-row launch (`Adapter::rows`, `units` unless the adapter
//!   says otherwise) against each row launched alone through the
//!   operation's one-row entry, for R up to the declared maximum.
//! - Batch: the probed units of a launch (`Adapter::batch`, rows unless the
//!   adapter says otherwise) against the same units beside other units'
//!   values.
//! - Ties and anchors: the adapter's tie and anchor-state pairs.
//!
//! Every buffer a compared launch writes starts poisoned, and the adapter
//! checks it after the launch (`written`, `written_over`): the live region
//! is written and finite, the rest untouched. Each adapter family holds a
//! CPU negative control for its write maps.
//!
//! An adapter may report that a run cannot qualify an entry on its host,
//! for example an entry of eight ranks on a host of four devices: the
//! harness then writes a receipt with `"qualified": false` and the reason,
//! lists the entry as untested and neither passes nor fails it.
//!
//! `ADAPTERS` lists which op kinds and properties have an adapter and in
//! which test binary; a CPU test fails when an entry claims a property its
//! op kind has no adapter for. `CHOPINFER_INVARIANCE_OPS` (comma-separated
//! op kinds, as `Gemm`), `CHOPINFER_INVARIANCE_FILTER` (symbol substrings,
//! any of which an entry must hold) and `CHOPINFER_INVARIANCE_SKIP`
//! (symbol substrings, none of which it may hold) narrow a GPU run, for
//! splitting it into jobs.

use kernel_core::contract::DType;
use kernel_lab::poison::{Elements, POISON, WriteMap};
use kernel_lab::{LabResult, Session};
use kernels_sm120::invariance::{Chunk, Declared, OpKind, Property, Rows, declarations};
use serde_json::json;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// Each launched unit's output bytes: a row, or an item's rows where the
/// adapter's unit is an item.
pub type Units = Vec<Vec<u8>>;

/// One bitwise comparison an adapter prepared.
pub struct Compared {
    pub label: String,
    pub got: Vec<u8>,
    pub expected: Vec<u8>,
}

/// How a compared buffer's elements are encoded, for the write checks.
/// (The comm binary checks no F32 values.)
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Element {
    Bf16,
    F32,
    /// Integers, IDs, codes or packed bytes of this many bytes: written
    /// means no longer the poison; there is no finite check.
    Raw(usize),
}

impl Element {
    pub fn bytes(self) -> usize {
        match self {
            Element::Bf16 => 2,
            Element::F32 => 4,
            Element::Raw(n) => n,
        }
    }

    /// The write-map element type: raw elements of 1, 4 and 8 bytes map as
    /// U8, I32 and I64, 2 bytes as BF16 values, 16 as 16-byte records.
    #[allow(dead_code)] // Not every binary's adapter maps by element.
    pub fn write_map(self) -> LabResult<Elements> {
        Ok(match self {
            Element::Bf16 | Element::Raw(2) => WriteMap::of(DType::Bf16),
            Element::F32 => WriteMap::of(DType::F32),
            Element::Raw(1) => WriteMap::of(DType::U8),
            Element::Raw(4) => WriteMap::of(DType::I32),
            Element::Raw(8) => WriteMap::of(DType::I64),
            Element::Raw(16) => WriteMap::of_records::<16>(),
            Element::Raw(n) => return Err(format!("no write-map element of {n} bytes")),
        })
    }

    fn finite(self, e: &[u8]) -> bool {
        match self {
            Element::Bf16 => u16::from_ne_bytes([e[0], e[1]]) & 0x7f80 != 0x7f80,
            Element::F32 => f32::from_ne_bytes([e[0], e[1], e[2], e[3]]).is_finite(),
            Element::Raw(_) => true,
        }
    }
}

/// Checks a buffer after a launch that found every byte of it poisoned
/// (`poison::POISON`): each element of `map`, the region the launch must
/// write (live rows, a route's slots), no longer holds the poison and, for
/// floating-point elements, is finite; every other byte (padding, rows
/// past the live ones) still holds the poison. Both arms of a comparison
/// would share an unwritten region's poison, or the NaN of arithmetic on
/// unwritten workspace, so equal bytes alone prove nothing.
pub fn written(label: &str, bytes: &[u8], map: &WriteMap, element: Element) -> Result<(), String> {
    let before = vec![POISON; bytes.len()];
    written_over(label, &before, bytes, map, element)
}

/// `written` for a buffer that holds data outside `map` before the launch
/// (a cache or state pool written in place): `before` is its image, whose
/// elements in `map` must hold the poison; after the launch each of them
/// is written (and finite), and every byte outside `map` kept its image.
pub fn written_over(
    label: &str,
    before: &[u8],
    after: &[u8],
    map: &WriteMap,
    element: Element,
) -> Result<(), String> {
    if map.element_bytes() != element.bytes() {
        return Err(format!(
            "{label}: a write map of {}-byte elements for {element:?}",
            map.element_bytes()
        ));
    }
    if before.len() != after.len() {
        return Err(format!(
            "{label}: {} bytes after the launch, {} before",
            after.len(),
            before.len()
        ));
    }
    if map.written_bytes() == 0 {
        return Err(format!("{label}: the launch must write something"));
    }
    if let Some(range) = map.ranges().last()
        && range.end > after.len()
    {
        return Err(format!(
            "{label}: write range {range:?} lies outside {} bytes",
            after.len()
        ));
    }
    let mut cursor = 0;
    for range in map.ranges() {
        unchanged(label, before, after, cursor..range.start)?;
        for at in range.clone().step_by(element.bytes()) {
            let (was, is) = (
                &before[at..at + element.bytes()],
                &after[at..at + element.bytes()],
            );
            let i = at / element.bytes();
            if was.iter().any(|&b| b != POISON) {
                return Err(format!(
                    "{label}: element {i} was not poisoned before the launch"
                ));
            }
            if is.iter().all(|&b| b == POISON) {
                return Err(format!(
                    "{label}: element {i} still holds the poison; the launch did not write it"
                ));
            }
            if !element.finite(is) {
                return Err(format!("{label}: element {i} is not finite ({is:02x?})"));
            }
        }
        cursor = range.end;
    }
    unchanged(label, before, after, cursor..after.len())
}

fn unchanged(
    label: &str,
    before: &[u8],
    after: &[u8],
    range: std::ops::Range<usize>,
) -> Result<(), String> {
    // One slice comparison first: in the opt-level 1 test build a byte loop
    // takes about 0.4 s per GiB, a slice comparison a tenth of that, and a
    // chained prefill layer's buffers hold 1.5 GB.
    if before[range.clone()] == after[range.clone()] {
        return Ok(());
    }
    match (range.clone()).find(|&at| before[at] != after[at]) {
        Some(at) => Err(format!(
            "{label}: byte {at} outside the written region changed from {:#04x} to {:#04x}",
            before[at], after[at]
        )),
        None => Ok(()),
    }
}

/// The op kinds and properties with an adapter, and the test binary that
/// holds it.
pub const ADAPTERS: &[(OpKind, &[Property], &str)] = &[
    (
        OpKind::Gemm,
        &[Property::Chunk, Property::Rows, Property::Batch],
        "ops",
    ),
    (
        OpKind::GroupedMoe,
        &[
            Property::Chunk,
            Property::Rows,
            Property::Batch,
            Property::Ties,
        ],
        "ops",
    ),
    (
        OpKind::Attention,
        &[Property::Chunk, Property::Rows, Property::Batch],
        "ops",
    ),
    (
        OpKind::SparseAttention,
        &[
            Property::Chunk,
            Property::Rows,
            Property::Batch,
            Property::Ties,
            Property::Anchors,
        ],
        "ops",
    ),
    (
        OpKind::Linear,
        &[
            Property::Chunk,
            Property::Rows,
            Property::Batch,
            Property::Anchors,
        ],
        "ops",
    ),
    (
        OpKind::Row,
        &[
            Property::Chunk,
            Property::Rows,
            Property::Batch,
            Property::Ties,
        ],
        "ops",
    ),
    (
        OpKind::Comm,
        &[Property::Chunk, Property::Rows, Property::Batch],
        "comm",
    ),
    (
        OpKind::Bundle,
        &[
            Property::Chunk,
            Property::Rows,
            Property::Batch,
            Property::Anchors,
        ],
        "ops",
    ),
    (
        OpKind::ContextParallel,
        &[Property::Chunk, Property::Rows, Property::Anchors],
        "ops",
    ),
    (
        OpKind::Sinkhorn,
        &[Property::Chunk, Property::Rows, Property::Batch],
        "ops",
    ),
    (
        OpKind::LatentAttention,
        &[Property::Chunk, Property::Rows, Property::Batch],
        "ops",
    ),
    (
        OpKind::PooledIndexer,
        &[
            Property::Chunk,
            Property::Rows,
            Property::Batch,
            Property::Ties,
            Property::Anchors,
        ],
        "ops",
    ),
    (
        OpKind::ChannelDecay,
        &[
            Property::Chunk,
            Property::Rows,
            Property::Batch,
            Property::Anchors,
        ],
        "channel_decay",
    ),
    (
        OpKind::ChainedMoe,
        &[
            Property::Chunk,
            Property::Rows,
            Property::Batch,
            Property::Ties,
        ],
        "moe_chained",
    ),
];

/// Every declaration: the kernel base's and every model's catalogs' (the
/// lab's `catalogs`), the lab's entries of families no catalog lists
/// (`kernel_lab::declared`), and the model bundles' own kernels'.
pub fn inventory() -> Vec<Declared> {
    let mut all = declarations(kernel_lab::catalogs());
    all.extend(kernel_lab::declared::lab_families());
    all.extend(bundle_mimo26::sm120::declared_kernels());
    all.extend(model_qwen38_contract::owned::declared());
    all
}

/// Whether an adapter tests `property` of `op` entries.
pub fn covered(op: OpKind, property: Property) -> bool {
    ADAPTERS
        .iter()
        .any(|(kind, properties, _)| *kind == op && properties.contains(&property))
}

/// Every declared claim whose op kind has no adapter for it.
pub fn uncovered() -> Vec<String> {
    inventory()
        .into_iter()
        .flat_map(|declared| {
            declared
                .invariance
                .claims()
                .into_iter()
                .filter(move |&property| !covered(declared.op, property))
                .map(move |property| format!("{} {:?} {property:?}", declared.symbol, declared.op))
        })
        .collect()
}

/// An op kind's launches. Each method serves one property; an adapter
/// that does not test a property keeps its default refusal.
pub trait Adapter {
    fn kind(&self) -> OpKind;

    /// Prompt rows a chunk case spans for `entry`.
    fn prompt_rows(&self, _entry: &Declared) -> u32 {
        512
    }

    /// Each row of a `rows`-row prompt prefilled in chunks ending at `ends`
    /// (ascending, the last is `rows`).
    fn chunks(
        &mut self,
        _sessions: &[Session],
        entry: &Declared,
        _rows: u32,
        _ends: &[u32],
    ) -> LabResult<Units> {
        Err(format!("{}: no chunk adapter", entry.symbol))
    }

    /// Each unit of one launch whose units are seeded by `seeds`.
    fn units(
        &mut self,
        _sessions: &[Session],
        entry: &Declared,
        _seeds: &[u64],
    ) -> LabResult<Units> {
        Err(format!("{}: no unit adapter", entry.symbol))
    }

    /// Each row of a rows case's launch of `seeds.len()` rows: `units`,
    /// unless an entry's rows case launches other units than its batch case
    /// (the QSA indexer's rows of one verify item, whose batch units are
    /// sequences).
    fn rows(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        self.units(sessions, entry, seeds)
    }

    /// Each unit of `seeds` launched alone through the operation's one-row
    /// entry.
    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        seeds
            .iter()
            .map(|&seed| {
                self.units(sessions, entry, &[seed])?
                    .pop()
                    .ok_or_else(|| format!("{}: no unit", entry.symbol))
            })
            .collect()
    }

    /// Row counts a rows case launches for `entry`'s one-row claim of `max`
    /// rows: by default `row_counts(max)`.
    fn row_counts(&self, _entry: &Declared, max: u32) -> Vec<u32> {
        row_counts(max)
    }

    /// Units a batch case launches beside the probed ones.
    fn batch_units(&self, _entry: &Declared) -> usize {
        4
    }

    /// Each unit of one batch case launch seeded by `seeds`: by default the
    /// units of `units`. An entry whose rows units read each other (the rows
    /// of one recurrent verify sequence) returns sequences here, since its
    /// batch claim is about the other sequences of a launch.
    fn batch(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        self.units(sessions, entry, seeds)
    }

    /// Why this run cannot qualify `entry`'s claims on this host (an entry
    /// of more ranks than the host has distinct devices), or `None`. The
    /// harness then records the entry as not qualified, never as tested.
    fn unqualified(&self, _sessions: &[Session], _entry: &Declared) -> Option<String> {
        None
    }

    fn ties(&mut self, _sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        Err(format!("{}: no tie adapter", entry.symbol))
    }

    fn anchors(&mut self, _sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        Err(format!("{}: no anchor adapter", entry.symbol))
    }
}

fn env_list(name: &str) -> Option<Vec<String>> {
    std::env::var(name)
        .ok()
        .map(|value| value.split(',').map(str::to_owned).collect())
}

/// The declared entries claiming `property` that `adapters` launch, after
/// the environment's narrowing.
pub fn entries(property: Property, kinds: &[OpKind]) -> Vec<Declared> {
    let ops = env_list("CHOPINFER_INVARIANCE_OPS");
    let filter = env_list("CHOPINFER_INVARIANCE_FILTER");
    let skip = env_list("CHOPINFER_INVARIANCE_SKIP");
    inventory()
        .into_iter()
        .filter(|d| d.invariance.claims().contains(&property) && kinds.contains(&d.op))
        .filter(|d| {
            ops.as_ref()
                .is_none_or(|ops| ops.iter().any(|op| format!("{:?}", d.op) == *op))
        })
        .filter(|d| {
            filter
                .as_ref()
                .is_none_or(|parts| parts.iter().any(|part| d.symbol.contains(part.as_str())))
        })
        .filter(|d| {
            skip.as_ref()
                .is_none_or(|parts| !parts.iter().any(|part| d.symbol.contains(part.as_str())))
        })
        .collect()
}

/// Prompt splits for a chunk case of `rows` rows at granule `g`: chunk ends,
/// each list ending at `rows`.
pub fn splits(rows: u32, g: u32) -> Vec<Vec<u32>> {
    let at = |n: u32| (n / g * g).clamp(g, rows - g);
    let mut splits = vec![
        vec![at(rows / 2), rows],
        vec![g, rows],
        vec![at(rows - g), rows],
        vec![
            at(rows / 3),
            at(rows / 3 + 5 * g),
            at(2 * rows / 3 + 3 * g),
            rows,
        ],
    ];
    if g == 1 {
        splits.push(vec![1, 17, 129, rows - 1, rows]);
    }
    for split in &mut splits {
        split.dedup();
    }
    splits.retain(|split| split.windows(2).all(|w| w[0] < w[1]) && split.len() > 1);
    splits
}

/// Row counts a rows case launches for a one-row claim of `max` rows: up
/// to eight rows, then 16, 32 and 64 where the claim reaches them.
pub fn row_counts(max: u32) -> Vec<u32> {
    if max == 1 {
        return vec![1];
    }
    let small = max.min(8);
    let mut counts = vec![2, small.div_ceil(2) + 1, small];
    counts.extend([16, 32, 64].into_iter().filter(|&n| n <= max));
    counts.retain(|&n| (2..=max).contains(&n));
    counts.dedup();
    counts
}

fn seeds(count: usize, salt: u64) -> Vec<u64> {
    (0..count as u64)
        .map(|i| 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(i + 1) ^ salt)
        .collect()
}

/// The comparisons of `property` for one entry.
pub(crate) fn compare(
    adapter: &mut dyn Adapter,
    sessions: &[Session],
    entry: &Declared,
    property: Property,
) -> LabResult<Vec<Compared>> {
    let mut compared = Vec::new();
    let mut units = |label: String, got: Units, expected: Units| -> LabResult<()> {
        if got.len() != expected.len() {
            return Err(format!(
                "{label}: {} units against {}",
                got.len(),
                expected.len()
            ));
        }
        for (i, (got, expected)) in got.into_iter().zip(expected).enumerate() {
            compared.push(Compared {
                label: format!("{label} unit {i}"),
                got,
                expected,
            });
        }
        Ok(())
    };
    match property {
        Property::Chunk => {
            let g = match entry.invariance.chunk {
                Chunk::Any => 1,
                Chunk::Granule(g) => g,
                Chunk::None => return Err("no chunk claim".into()),
            };
            let rows = adapter.prompt_rows(entry) / g * g;
            if rows == 0 {
                return Err(format!("a prompt shorter than the granule {g}"));
            }
            let whole = adapter.chunks(sessions, entry, rows, &[rows])?;
            // An entry of a few rows (a decode bucket that prefill steps
            // also run) splits into single granules; one granule repeats.
            let splits = if rows >= 4 * g {
                splits(rows, g)
            } else {
                vec![(g..=rows).step_by(g as usize).collect()]
            };
            for split in splits {
                let got = adapter.chunks(sessions, entry, rows, &split)?;
                units(format!("chunks {split:?}"), got, whole.clone())?;
            }
        }
        Property::Rows => {
            let Rows::OneRow { max } = entry.invariance.rows else {
                return Err("no rows claim".into());
            };
            for count in adapter.row_counts(entry, max) {
                let seeds = seeds(count as usize, u64::from(count));
                let got = adapter.rows(sessions, entry, &seeds)?;
                let alone = adapter.alone(sessions, entry, &seeds)?;
                units(format!("{count} rows"), got, alone)?;
            }
        }
        Property::Batch => {
            // An entry of a few rows holds at most its one-row maximum.
            let count = match entry.invariance.rows {
                Rows::OneRow { max } => adapter.batch_units(entry).min(max as usize),
                Rows::None => adapter.batch_units(entry),
            };
            if count < 2 {
                return Err(format!(
                    "a batch case needs two units, the entry holds {count}"
                ));
            }
            let first = seeds(count, 0xba7c);
            let mut other = first.clone();
            for (i, seed) in other.iter_mut().enumerate() {
                if i % 2 == 1 {
                    *seed ^= 0x005e_ed0f_07de;
                }
            }
            let a = adapter.batch(sessions, entry, &first)?;
            let b = adapter.batch(sessions, entry, &other)?;
            if a.len() != count || b.len() != count {
                return Err(format!(
                    "batch: {} and {} units of {count}",
                    a.len(),
                    b.len()
                ));
            }
            let even = |units: Units| units.into_iter().step_by(2).collect::<Units>();
            units("even units beside other odd units".into(), even(b), even(a))?;
        }
        Property::Ties => compared.extend(adapter.ties(sessions, entry)?),
        Property::Anchors => compared.extend(adapter.anchors(sessions, entry)?),
    }
    Ok(compared)
}

/// Runs the `property` harness over every declared entry `adapters`
/// launch: every comparison must be bitwise equal. Each entry writes a
/// receipt; failures are collected so one run reports them all.
pub fn run(
    property: Property,
    adapters: &mut [Box<dyn Adapter>],
    sessions: &[Session],
) -> LabResult<()> {
    let kinds = adapters.iter().map(|a| a.kind()).collect::<Vec<_>>();
    let entries = entries(property, &kinds);
    let mut failures = Vec::new();
    let mut unqualified = Vec::new();
    for entry in &entries {
        let adapter = adapters
            .iter_mut()
            .find(|a| a.kind() == entry.op)
            .expect("an adapter of the entry's kind");
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!(format!("invariance_{property:?}")));
        receipt.insert("symbol".into(), json!(entry.symbol));
        receipt.insert("op".into(), json!(format!("{:?}", entry.op)));
        receipt.insert(
            "invariance".into(),
            json!(format!("{:?}", entry.invariance)),
        );
        if let Some(reason) = adapter.unqualified(sessions, entry) {
            receipt.insert("qualified".into(), json!(false));
            receipt.insert("reason".into(), json!(reason));
            eprintln!(
                "{property:?} {}: not qualified on this host: {reason}",
                entry.symbol
            );
            unqualified.push(entry.symbol.clone());
            sessions[0].receipt("invariance", receipt)?;
            continue;
        }
        receipt.insert("qualified".into(), json!(true));
        let started = std::time::Instant::now();
        kernel_lab::take_launched();
        let outcome = compare(adapter.as_mut(), sessions, entry, property).and_then(|compared| {
            // A receipt is for an entry the case ran, not one its chain or
            // variant only names.
            let launched: BTreeSet<String> = kernel_lab::take_launched();
            if launched.contains(&entry.symbol) {
                Ok(compared)
            } else {
                Err(format!(
                    "the case never launched it (launched {})",
                    launched.into_iter().collect::<Vec<_>>().join(", ")
                ))
            }
        });
        receipt.insert("seconds".into(), json!(started.elapsed().as_secs_f64()));
        match outcome {
            Ok(compared) => {
                let differing = compared
                    .iter()
                    .filter(|c| c.got != c.expected)
                    .map(|c| {
                        let bytes = c
                            .got
                            .iter()
                            .zip(&c.expected)
                            .filter(|(a, b)| a != b)
                            .count()
                            + c.got.len().abs_diff(c.expected.len());
                        format!("{} ({bytes} of {} bytes)", c.label, c.expected.len())
                    })
                    .collect::<Vec<_>>();
                receipt.insert("comparisons".into(), json!(compared.len()));
                receipt.insert("differing".into(), json!(differing));
                eprintln!(
                    "{property:?} {}: {} comparisons, {} differ",
                    entry.symbol,
                    compared.len(),
                    differing.len()
                );
                // A case that compares nothing fails, and its receipt says
                // so rather than reading as a pass.
                let vacuous = if compared.is_empty() {
                    Some("no comparison".to_owned())
                } else {
                    compared
                        .iter()
                        .find(|c| c.expected.is_empty())
                        .map(|empty| format!("{} compares no bytes", empty.label))
                };
                if let Some(error) = vacuous {
                    receipt.insert("error".into(), json!(error));
                    failures.push(format!("{}: {error}", entry.symbol));
                } else if !differing.is_empty() {
                    failures.push(format!("{}: {}", entry.symbol, differing.join(", ")));
                }
            }
            Err(error) => {
                receipt.insert("error".into(), json!(error));
                eprintln!("{property:?} {}: {error}", entry.symbol);
                failures.push(format!("{}: {error}", entry.symbol));
            }
        }
        sessions[0].receipt("invariance", receipt)?;
    }
    let tested = entries.len() - unqualified.len();
    if tested == 0 {
        return Err(format!(
            "no declared {property:?} entry for {kinds:?} this run can qualify"
        ));
    }
    if !unqualified.is_empty() {
        eprintln!(
            "{property:?}: {} entries not qualified on this host (untested): {}",
            unqualified.len(),
            unqualified.join(", ")
        );
    }
    if failures.is_empty() {
        eprintln!("{property:?}: {tested} entries hold");
        Ok(())
    } else {
        Err(format!(
            "{} of {tested} {property:?} entries fail: {}",
            failures.len(),
            failures.join("; ")
        ))
    }
}
