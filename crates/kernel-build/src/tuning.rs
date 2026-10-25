//! Kernel schedule tuning tables.
//!
//! One table format for every tunable operation. A row is keyed by the
//! operation, the weight format, the rank shape (the derived per-rank sizes
//! the instance is compiled for), the batch bucket and an optional launch-row
//! range, and holds the schedule and its compile-time knobs. A catalog
//! builds each instance from its row, so a new model or topology needs rows,
//! not kernel code.
//!
//! The file is canonical: rows in ascending key order, one row per line,
//! knobs and shape entries in name order. [`Table::parse`] refuses anything
//! else and [`Table::render`] writes exactly that form, so an offline sweep
//! that rewrites a table produces a minimal, reviewable diff.
//!
//! This module interprets no knob. The owning catalog validates a row's
//! schedule and knobs against its kernel; [`accepts`] is the shared rule an
//! offline sweep uses to replace a row.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::fmt;

/// Schema identifier of the table file.
pub const SCHEMA: &str = "chopinfer-tuning-v1";

/// A malformed or non-canonical table, or a missing row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningError(pub String);

impl fmt::Display for TuningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for TuningError {}

fn error(message: impl Into<String>) -> TuningError {
    TuningError(message.into())
}

/// One compile-time knob value: a count or a flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(untagged)]
pub enum Knob {
    Flag(bool),
    Count(u32),
}

impl Knob {
    pub fn count(self) -> Option<u32> {
        match self {
            Knob::Count(value) => Some(value),
            Knob::Flag(_) => None,
        }
    }

    pub fn flag(self) -> Option<bool> {
        match self {
            Knob::Flag(value) => Some(value),
            Knob::Count(_) => None,
        }
    }
}

impl fmt::Display for Knob {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Knob::Flag(value) => write!(f, "{value}"),
            Knob::Count(value) => write!(f, "{value}"),
        }
    }
}

/// Knobs by name.
pub type Knobs = BTreeMap<String, Knob>;

/// What a row is for. Keys order by operation, weights, shape, bucket, then rows.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Key {
    /// Dotted operation name, e.g. `grouped_moe.gate_up.decode`.
    pub op: String,
    /// Weight format, e.g. `mxfp4`.
    pub weights: String,
    /// Per-rank sizes the instance is compiled for, by name.
    pub shape: BTreeMap<String, u32>,
    /// Batch bucket: the token capacity of the instance.
    pub bucket: u32,
    /// Inclusive launch-row range within the bucket; absent covers `1..=bucket`.
    pub rows: Option<[u32; 2]>,
}

impl Key {
    fn group(&self) -> (&str, &str, &BTreeMap<String, u32>, u32) {
        (&self.op, &self.weights, &self.shape, self.bucket)
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} ", self.op, self.weights)?;
        for (i, (name, value)) in self.shape.iter().enumerate() {
            write!(f, "{}{name}={value}", if i == 0 { "" } else { "," })?;
        }
        write!(f, " t{}", self.bucket)?;
        if let Some([lo, hi]) = self.rows {
            write!(f, " rows={lo}..={hi}")?;
        }
        Ok(())
    }
}

/// The schedule and knobs of one key.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    #[serde(flatten)]
    pub key: Key,
    /// Schedule name, e.g. `ring`.
    pub schedule: String,
    pub knobs: Knobs,
}

/// The device the rows were tuned on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub compute_capability: u32,
    pub sm_count: u32,
}

/// A tuning table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    pub target: Target,
    rows: Vec<Row>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    schema: String,
    target: Target,
    rows: Vec<Row>,
}

/// Lowercase ASCII letters, digits, `_` and `.` separators, starting with a
/// letter.
fn name(value: &str) -> bool {
    value.starts_with(|c: char| c.is_ascii_lowercase())
        && value
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '.')
}

fn check_row(row: &Row) -> Result<(), TuningError> {
    let key = &row.key;
    if !name(&key.op) || !name(&key.weights) || !name(&row.schedule) {
        return Err(error(format!(
            "{key}: invalid op, weights or schedule name"
        )));
    }
    if key.shape.is_empty() || key.shape.keys().any(|n| !name(n)) {
        return Err(error(format!("{key}: shape needs named sizes")));
    }
    if key.bucket == 0 {
        return Err(error(format!("{key}: bucket must be nonzero")));
    }
    if let Some([lo, hi]) = key.rows {
        if lo == 0 || lo > hi || hi > key.bucket {
            return Err(error(format!(
                "{key}: rows must satisfy 1 <= lo <= hi <= bucket"
            )));
        }
    }
    if row.knobs.keys().any(|n| !name(n)) {
        return Err(error(format!("{key}: knobs need names")));
    }
    Ok(())
}

fn check_ranges(left: &Key, right: &Key) -> Result<(), TuningError> {
    if left.group() == right.group() {
        match (left.rows, right.rows) {
            (Some([_, hi]), Some([lo, _])) if hi < lo => {}
            (Some(_), Some(_)) => {
                return Err(error(format!(
                    "overlapping tuning row ranges: {left} then {right}"
                )));
            }
            _ => {
                return Err(error(format!(
                    "whole-bucket tuning row must stand alone: {left} then {right}"
                )));
            }
        }
    }
    Ok(())
}

impl Table {
    /// An empty table for `target`.
    pub fn new(target: Target) -> Self {
        Self {
            target,
            rows: Vec::new(),
        }
    }

    /// Parses a canonical table: the schema, valid names, strictly
    /// ascending keys with disjoint ranges, and bytes equal to
    /// [`Table::render`] of the result.
    pub fn parse(text: &str) -> Result<Self, TuningError> {
        let file: File =
            serde_json::from_str(text).map_err(|e| error(format!("tuning table: {e}")))?;
        if file.schema != SCHEMA {
            return Err(error(format!(
                "tuning table schema {:?}, expected {SCHEMA:?}",
                file.schema
            )));
        }
        for row in &file.rows {
            check_row(row)?;
        }
        if let Some(pair) = file.rows.windows(2).find(|p| p[0].key >= p[1].key) {
            return Err(error(format!(
                "tuning rows must be in ascending key order without repeats: {} then {}",
                pair[0].key, pair[1].key
            )));
        }
        for pair in file.rows.windows(2) {
            check_ranges(&pair[0].key, &pair[1].key)?;
        }
        let table = Self {
            target: file.target,
            rows: file.rows,
        };
        if table.render() != text {
            return Err(error(
                "tuning table is not in canonical form; rewrite it with Table::render",
            ));
        }
        Ok(table)
    }

    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    pub fn get(&self, key: &Key) -> Option<&Row> {
        self.rows
            .binary_search_by(|row| row.key.cmp(key))
            .ok()
            .map(|at| &self.rows[at])
    }

    /// The row for this exact operation, format, shape and bucket whose
    /// inclusive range contains `rows`. Gaps and rows outside the bucket
    /// return `None`; a row without a range covers `1..=bucket`.
    pub fn select(
        &self,
        op: &str,
        weights: &str,
        shape: &BTreeMap<String, u32>,
        bucket: u32,
        rows: u32,
    ) -> Option<&Row> {
        if rows == 0 || rows > bucket {
            return None;
        }
        self.rows
            .binary_search_by(|row| {
                row.key
                    .group()
                    .cmp(&(op, weights, shape, bucket))
                    .then_with(|| {
                        let [lo, hi] = row.key.rows.unwrap_or([1, bucket]);
                        if rows < lo {
                            std::cmp::Ordering::Greater
                        } else if rows > hi {
                            std::cmp::Ordering::Less
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    })
            })
            .ok()
            .map(|at| &self.rows[at])
    }

    /// The row of `key`, or an error naming it.
    pub fn require(&self, key: &Key) -> Result<&Row, TuningError> {
        self.get(key)
            .ok_or_else(|| error(format!("no tuning row for {key}")))
    }

    /// Inserts `row`, replacing the row of the same key. Returns the
    /// replaced row. Refuses overlapping or mixed whole-bucket/ranged rows
    /// without changing the table.
    pub fn upsert(&mut self, row: Row) -> Result<Option<Row>, TuningError> {
        check_row(&row)?;
        match self.rows.binary_search_by(|r| r.key.cmp(&row.key)) {
            Ok(at) => Ok(Some(std::mem::replace(&mut self.rows[at], row))),
            Err(at) => {
                if at > 0 {
                    check_ranges(&self.rows[at - 1].key, &row.key)?;
                }
                if let Some(next) = self.rows.get(at) {
                    check_ranges(&row.key, &next.key)?;
                }
                self.rows.insert(at, row);
                Ok(None)
            }
        }
    }

    /// The canonical file text.
    pub fn render(&self) -> String {
        let quote = |s: &str| serde_json::to_string(s).expect("string serializes");
        let mut out = format!(
            "{{\n  \"schema\": {},\n  \"target\": {{\"compute_capability\": {}, \"sm_count\": {}}},\n  \"rows\": [",
            quote(SCHEMA),
            self.target.compute_capability,
            self.target.sm_count
        );
        for (i, row) in self.rows.iter().enumerate() {
            let shape = row
                .key
                .shape
                .iter()
                .map(|(n, v)| format!("{}: {v}", quote(n)))
                .collect::<Vec<_>>()
                .join(", ");
            let knobs = row
                .knobs
                .iter()
                .map(|(n, v)| format!("{}: {v}", quote(n)))
                .collect::<Vec<_>>()
                .join(", ");
            let rows = row
                .key
                .rows
                .map_or_else(String::new, |[lo, hi]| format!(", \"rows\": [{lo}, {hi}]"));
            out.push_str(if i == 0 { "\n" } else { ",\n" });
            out.push_str(&format!(
                "    {{\"op\": {}, \"weights\": {}, \"shape\": {{{shape}}}, \"bucket\": {}{rows}, \"schedule\": {}, \"knobs\": {{{knobs}}}}}",
                quote(&row.key.op),
                quote(&row.key.weights),
                row.key.bucket,
                quote(&row.schedule),
            ));
        }
        out.push_str(if self.rows.is_empty() {
            "]\n}\n"
        } else {
            "\n  ]\n}\n"
        });
        out
    }
}

/// Orders the tuned sources a geometry may inherit from: the geometry
/// itself (`own`) first, then by increasing `distance` (the model's chosen
/// L1 distance between shapes), then by id. Equal keys keep their order.
pub fn nearest_first<'a, T>(
    own: Option<&str>,
    sources: impl IntoIterator<Item = T>,
    id: impl Fn(&T) -> &'a str,
    distance: impl Fn(&T) -> u64,
) -> Vec<T> {
    let mut sources = sources.into_iter().collect::<Vec<_>>();
    sources.sort_by_key(|source| (own != Some(id(source)), distance(source), id(source)));
    sources
}

/// Where [`build_nearest`] found the table it built from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TableSource<'a> {
    /// The geometry's own table.
    Own,
    /// The nearest other geometry's table that builds, by ID.
    Inherited(&'a str),
    /// The engine defaults.
    Default,
}

/// Builds a geometry's entries from its own table in `tables` (by geometry
/// ID), which must build, else from the first other table that builds in
/// [`nearest_first`] order by `distance` (of a table's ID), else from the
/// engine's `default`. `what` names the tables in refusals.
pub fn build_nearest<'a, T, V>(
    own: &str,
    what: &str,
    tables: &'a BTreeMap<String, T>,
    distance: impl Fn(&str) -> u64,
    default: impl FnOnce() -> T,
    build: impl Fn(&T) -> Result<V, String>,
) -> Result<(V, TableSource<'a>), String> {
    if let Some(table) = tables.get(own) {
        return build(table)
            .map(|built| (built, TableSource::Own))
            .map_err(|error| format!("geometry {own} measured {what} table refused: {error}"));
    }
    let mut refusals = Vec::new();
    for (id, table) in nearest_first(
        Some(own),
        tables,
        |(id, _)| id.as_str(),
        |(id, _)| distance(id),
    ) {
        match build(table) {
            Ok(built) => return Ok((built, TableSource::Inherited(id))),
            Err(error) => refusals.push(format!("{id}: {error}")),
        }
    }
    build(&default())
        .map(|built| (built, TableSource::Default))
        .map_err(|error| {
            format!(
                "geometry {own} {what} fallback refused: {}; engine defaults: {error}",
                refusals.join("; ")
            )
        })
}

/// The L1 distance from `key`'s shape to `other`'s over `key`'s sizes.
///
/// Panics if `other` lacks one of them.
pub fn shape_distance(key: &Key, other: &Key) -> u64 {
    key.shape
        .iter()
        .map(|(name, value)| u64::from(value.abs_diff(other.shape[name])))
        .sum()
}

/// The row an untuned `key` inherits: among the rows of `tables` (by
/// geometry id) for the same operation, weights and bucket, the first in
/// [`nearest_first`] order by [`shape_distance`] that is `legal` once keyed
/// to `key`, with the status `inherited <id> row <source key>`. `own` is the
/// geometry `key` belongs to.
pub fn inherit<'a>(
    own: Option<&str>,
    key: &Key,
    tables: impl IntoIterator<Item = (&'a str, &'a Table)>,
    mut legal: impl FnMut(&Row) -> bool,
) -> Option<(Row, String)> {
    let rows = tables
        .into_iter()
        .flat_map(|(id, table)| table.rows().iter().map(move |row| (id, row)))
        .filter(|(_, row)| {
            row.key.op == key.op && row.key.weights == key.weights && row.key.bucket == key.bucket
        });
    nearest_first(
        own,
        rows,
        |(id, _)| id,
        |(_, row)| shape_distance(key, &row.key),
    )
    .into_iter()
    .find_map(|(id, row)| {
        let status = format!("inherited {id} row {}", row.key);
        let row = Row {
            key: key.clone(),
            ..row.clone()
        };
        legal(&row).then_some((row, status))
    })
}

/// Timing of a candidate against the current row, from alternating
/// A B A B repetitions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Comparison {
    /// Median candidate time over median current-row time.
    pub ratio: f64,
    /// Widest relative difference between the two A timings of one
    /// repetition: the noise band of this measurement.
    pub noise: f64,
}

/// The replacement rule: a candidate replaces the current row only when it
/// is faster by more than the noise band plus `margin`.
pub fn accepts(comparison: Comparison, margin: f64) -> bool {
    comparison.ratio.is_finite() && comparison.ratio < 1.0 - (comparison.noise + margin)
}

/// The candidate to confirm, among bitwise-equal ones that pass [`accepts`]:
/// the lowest ratio, ties broken by ascending knobs. `None` keeps the
/// current row.
pub fn best<'a>(
    candidates: impl IntoIterator<Item = (&'a Knobs, Comparison)>,
    margin: f64,
) -> Option<&'a Knobs> {
    candidates
        .into_iter()
        .filter(|(_, c)| accepts(*c, margin))
        .min_by(|a, b| a.1.ratio.total_cmp(&b.1.ratio).then_with(|| a.0.cmp(b.0)))
        .map(|(knobs, _)| knobs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(op: &str, bucket: u32) -> Key {
        Key {
            op: op.into(),
            weights: "mxfp4".into(),
            shape: [("experts".to_string(), 64), ("hidden".to_string(), 4096)].into(),
            bucket,
            rows: None,
        }
    }

    fn row(op: &str, bucket: u32, warps: u32) -> Row {
        Row {
            key: key(op, bucket),
            schedule: "ring".into(),
            knobs: [
                ("warps".to_string(), Knob::Count(warps)),
                ("early_row".to_string(), Knob::Flag(true)),
            ]
            .into(),
        }
    }

    fn table() -> Table {
        let mut table = Table::new(Target {
            compute_capability: 120,
            sm_count: 188,
        });
        for (op, bucket) in [("b.op", 2), ("a.op", 16), ("a.op", 2)] {
            table.upsert(row(op, bucket, 8)).unwrap();
        }
        table
    }

    #[test]
    fn render_is_canonical_and_parses_back() {
        let table = table();
        let text = table.render();
        assert_eq!(
            text,
            "{\n  \"schema\": \"chopinfer-tuning-v1\",\n  \"target\": {\"compute_capability\": 120, \"sm_count\": 188},\n  \"rows\": [\n    \
             {\"op\": \"a.op\", \"weights\": \"mxfp4\", \"shape\": {\"experts\": 64, \"hidden\": 4096}, \"bucket\": 2, \"schedule\": \"ring\", \"knobs\": {\"early_row\": true, \"warps\": 8}},\n    \
             {\"op\": \"a.op\", \"weights\": \"mxfp4\", \"shape\": {\"experts\": 64, \"hidden\": 4096}, \"bucket\": 16, \"schedule\": \"ring\", \"knobs\": {\"early_row\": true, \"warps\": 8}},\n    \
             {\"op\": \"b.op\", \"weights\": \"mxfp4\", \"shape\": {\"experts\": 64, \"hidden\": 4096}, \"bucket\": 2, \"schedule\": \"ring\", \"knobs\": {\"early_row\": true, \"warps\": 8}}\n  ]\n}\n"
        );
        assert_eq!(Table::parse(&text).unwrap(), table);
        let empty = Table::new(table.target);
        assert_eq!(Table::parse(&empty.render()).unwrap(), empty);
    }

    #[test]
    fn schedule_only_rows_allow_empty_knobs() {
        let mut table = Table::new(table().target);
        let mut choice = row("a.op", 16, 4);
        choice.knobs.clear();
        table.upsert(choice.clone()).unwrap();
        assert_eq!(Table::parse(&table.render()).unwrap().rows(), &[choice]);
    }

    #[test]
    fn parse_and_render_inclusive_row_ranges() {
        let text = table()
            .render()
            .replace("\"bucket\": 16,", "\"bucket\": 16, \"rows\": [3, 12],");
        assert_eq!(Table::parse(&text).unwrap().render(), text);
    }

    fn ranged(rows: Option<[u32; 2]>) -> Row {
        Row {
            key: Key {
                rows,
                ..key("a.op", 16)
            },
            ..row("a.op", 16, 4)
        }
    }

    #[test]
    fn parse_and_upsert_refuse_invalid_ranges_and_knob_names() {
        for range in [[0, 1], [5, 4], [1, 17], [17, 17]] {
            let row = ranged(Some(range));
            let mut table = Table::new(table().target);
            assert!(table.upsert(row.clone()).unwrap_err().0.contains("1 <= lo"));
            assert!(table.rows().is_empty());
            table.rows.push(row);
            assert!(
                Table::parse(&table.render())
                    .unwrap_err()
                    .0
                    .contains("1 <= lo")
            );
        }
        let mut table = Table::new(table().target);
        let mut row = ranged(None);
        row.knobs = [("Bad knob".into(), Knob::Count(4))].into();
        assert!(
            table
                .upsert(row.clone())
                .unwrap_err()
                .0
                .contains("knobs need names")
        );
        assert!(table.rows().is_empty());
        table.rows.push(row);
        assert!(
            Table::parse(&table.render())
                .unwrap_err()
                .0
                .contains("knobs need names")
        );
    }

    #[test]
    fn parse_refuses_overlaps_and_mixed_ranges_in_ascending_key_order() {
        for (left, right, message) in [
            (Some([1, 8]), Some([8, 16]), "overlapping"),
            (Some([1, 16]), Some([3, 4]), "overlapping"),
            (Some([1, 4]), Some([1, 8]), "overlapping"),
            (None, Some([3, 4]), "stand alone"),
        ] {
            let table = Table {
                target: table().target,
                rows: vec![ranged(left), ranged(right)],
            };
            assert!(table.rows[0].key < table.rows[1].key);
            assert!(
                Table::parse(&table.render())
                    .unwrap_err()
                    .0
                    .contains(message)
            );
        }
        let mut table = Table {
            target: table().target,
            rows: vec![ranged(Some([1, 4])), ranged(Some([5, 8]))],
        };
        assert_eq!(Table::parse(&table.render()).unwrap(), table);
        table.rows.reverse();
        assert!(
            Table::parse(&table.render())
                .unwrap_err()
                .0
                .contains("ascending")
        );
        table.rows[1] = table.rows[0].clone();
        assert!(
            Table::parse(&table.render())
                .unwrap_err()
                .0
                .contains("ascending")
        );
    }

    #[test]
    fn upsert_refuses_range_conflicts_without_mutating_the_table() {
        let mut table = Table::new(table().target);
        for range in [[12, 16], [1, 4], [8, 10]] {
            table.upsert(ranged(Some(range))).unwrap();
        }
        for range in [
            None,
            Some([1, 3]),
            Some([4, 7]),
            Some([5, 8]),
            Some([9, 11]),
            Some([11, 12]),
            Some([1, 16]),
        ] {
            let before = table.clone();
            assert!(table.upsert(ranged(range)).is_err());
            assert_eq!(table, before);
        }
        table.upsert(ranged(Some([5, 7]))).unwrap();
        table.upsert(ranged(Some([11, 11]))).unwrap();
        let mut replacement = ranged(Some([8, 10]));
        replacement.schedule = "ticket".into();
        assert_eq!(
            table.upsert(replacement.clone()).unwrap(),
            Some(ranged(Some([8, 10])))
        );
        assert_eq!(table.require(&replacement.key).unwrap(), &replacement);
        assert_eq!(Table::parse(&table.render()).unwrap(), table);

        let mut whole = Table::new(table.target);
        whole.upsert(ranged(None)).unwrap();
        let before = whole.clone();
        assert!(whole.upsert(ranged(Some([1, 16]))).is_err());
        assert_eq!(whole, before);
    }

    #[test]
    fn select_matches_only_the_exact_group_and_inclusive_range() {
        let mut table = Table::new(table().target);
        for range in [[3, 5], [9, 12], [13, 16]] {
            table.upsert(ranged(Some(range))).unwrap();
        }
        let key = key("a.op", 16);
        for rows in 0..=17 {
            let expected = match rows {
                3..=5 => Some([3, 5]),
                9..=12 => Some([9, 12]),
                13..=16 => Some([13, 16]),
                _ => None,
            };
            assert_eq!(
                table
                    .select(&key.op, &key.weights, &key.shape, key.bucket, rows)
                    .map(|r| r.key.rows.unwrap()),
                expected
            );
        }
        assert!(table.get(&key).is_none());
        assert!(table.require(&key).is_err());
        let mut others = Vec::new();
        let mut other = ranged(None);
        other.key.op = "b.op".into();
        others.push(other);
        let mut other = ranged(None);
        other.key.weights = "mxfp8".into();
        others.push(other);
        let mut other = ranged(None);
        other.key.shape.insert("hidden".into(), 8192);
        others.push(other);
        let mut other = ranged(None);
        other.key.bucket = 32;
        others.push(other);
        for other in others {
            let key = &other.key;
            assert!(
                table
                    .select(&key.op, &key.weights, &key.shape, key.bucket, 3)
                    .is_none()
            );
            table.upsert(other.clone()).unwrap();
            for rows in 0..=key.bucket + 1 {
                let expected = (1..=key.bucket).contains(&rows).then_some(&other);
                assert_eq!(
                    table.select(&key.op, &key.weights, &key.shape, key.bucket, rows),
                    expected
                );
            }
            assert_eq!(table.get(key), Some(&other));
        }
        assert!(
            table
                .select(&key.op, &key.weights, &key.shape, key.bucket, 1)
                .is_none()
        );
    }

    #[test]
    fn existing_tuning_v1_tables_keep_every_byte() {
        for (name, text) in [
            (
                "qwen",
                include_str!("../../../models/qwen38/contract/tuning/flash/schedules.json"),
            ),
            (
                "glm",
                include_str!("../../../models/glm5next/contract/tuning/flash/schedules.json"),
            ),
            (
                "sm120",
                include_str!("../../kernels-sm120/tuning/schedules.json"),
            ),
        ] {
            let table = Table::parse(text).unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(table.rows().is_empty(), name == "sm120", "{name}");
            assert!(
                table.rows().iter().all(|row| row.key.rows.is_none()),
                "{name}"
            );
            assert_eq!(table.render().as_bytes(), text.as_bytes(), "{name}");
        }

        // MiMo wraps its canonical tables in a map keyed by geometry.
        let text = include_str!("../../../models/mimo26/contract/tuning/schedules.json");
        let files: BTreeMap<String, File> = serde_json::from_str(text).unwrap();
        assert!(!files.is_empty());
        let mut entries = Vec::new();
        for (id, file) in files {
            assert_eq!(file.schema, SCHEMA);
            let table = Table {
                target: file.target,
                rows: file.rows,
            };
            assert!(!table.rows().is_empty(), "{id}: expected existing rows");
            assert!(
                table.rows().iter().all(|row| row.key.rows.is_none()),
                "{id}"
            );
            let rendered = table.render();
            assert_eq!(Table::parse(&rendered).unwrap(), table);
            let indented = rendered.trim_end_matches('\n').replace('\n', "\n  ");
            entries.push(format!(
                "  {}: {indented}",
                serde_json::to_string(&id).unwrap()
            ));
        }
        let rendered = format!("{{\n{}\n}}\n", entries.join(",\n"));
        assert_eq!(rendered.as_bytes(), text.as_bytes(), "mimo");
    }

    #[test]
    fn parse_refuses_non_canonical_tables() {
        let text = table().render();
        // Whitespace, order, repeats, schema and unknown fields.
        assert!(Table::parse(&text.replace(", \"bucket\"", ",\"bucket\"")).is_err());
        let lines: Vec<&str> = text.lines().collect();
        let swapped = [
            &lines[..3],
            &[lines[4].trim_end_matches(','), &format!("{},", lines[3])][..],
            &lines[5..],
        ]
        .concat()
        .join("\n");
        assert!(Table::parse(&(swapped + "\n")).is_err());
        let repeated = text.replace("\"bucket\": 16", "\"bucket\": 2");
        assert!(Table::parse(&repeated).unwrap_err().0.contains("ascending"));
        assert!(Table::parse(&text.replace("tuning-v1", "tuning-v0")).is_err());
        assert!(Table::parse(&text.replace("\"schedule\"", "\"extra\": 1, \"schedule\"")).is_err());
        assert!(Table::parse(&text.replace("\"a.op\"", "\"A op\"")).is_err());
    }

    #[test]
    fn lookup_and_upsert_by_key() {
        let mut table = table();
        assert_eq!(
            table.get(&key("a.op", 16)).unwrap().knobs["warps"],
            Knob::Count(8)
        );
        assert!(table.get(&key("a.op", 4)).is_none());
        assert!(
            table
                .require(&key("a.op", 4))
                .unwrap_err()
                .0
                .contains("a.op mxfp4 experts=64,hidden=4096 t4")
        );
        let old = table.upsert(row("a.op", 16, 4)).unwrap().unwrap();
        assert_eq!(old.knobs["warps"], Knob::Count(8));
        assert_eq!(
            table.get(&key("a.op", 16)).unwrap().knobs["warps"],
            Knob::Count(4)
        );
        assert_eq!(table.rows().len(), 3);
        assert!(table.upsert(row("a.op", 0, 4)).is_err());
    }

    #[test]
    fn an_untuned_key_inherits_the_nearest_legal_row() {
        let sized = |op: &str, hidden: u32, warps: u32| Row {
            key: Key {
                shape: [("experts".to_string(), 64), ("hidden".to_string(), hidden)].into(),
                ..key(op, 2)
            },
            ..row(op, 2, warps)
        };
        let table_of = |rows: &[Row]| {
            let mut table = Table::new(table().target);
            for row in rows {
                table.upsert(row.clone()).unwrap();
            }
            table
        };
        let (far, near, other) = (
            table_of(&[sized("a.op", 1024, 8)]),
            table_of(&[sized("a.op", 3072, 4), sized("b.op", 4096, 2)]),
            table_of(&[sized("a.op", 4096, 16)]),
        );
        let tables = [("far", &far), ("near", &near), ("own", &other)];
        let want = key("a.op", 2);
        // The geometry's own row comes first, then the nearest shape.
        let (row, status) = inherit(Some("own"), &want, tables, |_| true).unwrap();
        assert_eq!(
            (row.key.clone(), row.knobs["warps"]),
            (want.clone(), Knob::Count(16))
        );
        assert_eq!(
            status,
            "inherited own row a.op mxfp4 experts=64,hidden=4096 t2"
        );
        let (row, status) = inherit(None, &want, tables, |_| true).unwrap();
        assert_eq!(row.knobs["warps"], Knob::Count(16));
        assert!(status.starts_with("inherited own row"));
        let (row, status) = inherit(None, &want, [tables[0], tables[1]], |_| true).unwrap();
        assert_eq!(row.knobs["warps"], Knob::Count(4));
        assert_eq!(
            status,
            "inherited near row a.op mxfp4 experts=64,hidden=3072 t2"
        );
        // An illegal row is skipped; nothing legal inherits nothing.
        let legal = |row: &Row| row.knobs["warps"] != Knob::Count(4);
        let (row, _) = inherit(None, &want, [tables[0], tables[1]], legal).unwrap();
        assert_eq!(row.knobs["warps"], Knob::Count(8));
        assert!(inherit(None, &want, [tables[0], tables[1]], |_| false).is_none());
        assert!(inherit(None, &key("c.op", 2), tables, |_| true).is_none());
        // Equal distances order by id; the own geometry precedes both.
        let ids = nearest_first(Some("b"), ["c", "a", "b"], |id| id, |_| 1);
        assert_eq!(ids, ["b", "a", "c"]);
    }

    /// A geometry builds from its own table or refuses; without one, from
    /// the nearest table that builds, then from the defaults.
    #[test]
    fn tables_build_from_the_own_then_the_nearest_then_the_defaults() {
        let tables: BTreeMap<String, i32> =
            [("a".into(), 1), ("b".into(), -2), ("c".into(), 3)].into();
        let build = |t: &i32| {
            if *t > 0 {
                Ok(*t * 10)
            } else {
                Err(format!("{t} refused"))
            }
        };
        let distance = |id: &str| u64::from(id != "b");
        let nearest = |own| build_nearest(own, "test", &tables, distance, || 7, build);
        assert_eq!(nearest("a"), Ok((10, TableSource::Own)));
        assert_eq!(
            nearest("b"),
            Err("geometry b measured test table refused: -2 refused".into())
        );
        assert_eq!(nearest("d"), Ok((10, TableSource::Inherited("a"))));
        let refused: BTreeMap<String, i32> = [("b".into(), -2)].into();
        assert_eq!(
            build_nearest("d", "test", &refused, distance, || 7, build),
            Ok((70, TableSource::Default))
        );
        assert_eq!(
            build_nearest("d", "test", &refused, distance, || -1, build),
            Err(
                "geometry d test fallback refused: b: -2 refused; engine defaults: -1 refused"
                    .into()
            )
        );
    }

    #[test]
    fn replacement_needs_a_margin_above_noise_and_ties_keep_knob_order() {
        let c = |ratio, noise| Comparison { ratio, noise };
        assert!(accepts(c(0.97, 0.01), 0.01));
        assert!(!accepts(c(0.98, 0.01), 0.01));
        assert!(!accepts(c(0.95, 0.04), 0.01));
        assert!(!accepts(c(f64::NAN, 0.0), 0.01));
        let knobs = |warps| -> Knobs { [("warps".to_string(), Knob::Count(warps))].into() };
        let (four, eight, sixteen) = (knobs(4), knobs(8), knobs(16));
        assert_eq!(best([(&eight, c(0.99, 0.0))], 0.01), None);
        let chosen = best(
            [
                (&sixteen, c(0.90, 0.0)),
                (&eight, c(0.90, 0.0)),
                (&four, c(0.95, 0.0)),
            ],
            0.01,
        );
        assert_eq!(chosen, Some(&eight));
    }
}
