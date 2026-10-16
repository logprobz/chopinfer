//! Kernel schedule tuning tables.
//!
//! One table format for every tunable operation. A row is keyed by the
//! operation, the weight format, the rank shape (the derived per-rank sizes
//! the instance is compiled for) and the batch bucket, and holds the
//! schedule and its compile-time knobs. A catalog builds each instance from
//! its row, so a new model or topology needs rows, not kernel code.
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

/// What a row is for. Keys order by operation, weights, shape, then bucket.
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
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} ", self.op, self.weights)?;
        for (i, (name, value)) in self.shape.iter().enumerate() {
            write!(f, "{}{name}={value}", if i == 0 { "" } else { "," })?;
        }
        write!(f, " t{}", self.bucket)
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
    if row.knobs.is_empty() || row.knobs.keys().any(|n| !name(n)) {
        return Err(error(format!("{key}: knobs need names")));
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
    /// ascending keys, and bytes equal to [`Table::render`] of the result.
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

    /// The row of `key`, or an error naming it.
    pub fn require(&self, key: &Key) -> Result<&Row, TuningError> {
        self.get(key)
            .ok_or_else(|| error(format!("no tuning row for {key}")))
    }

    /// Inserts `row`, replacing the row of the same key. Returns the
    /// replaced row.
    pub fn upsert(&mut self, row: Row) -> Result<Option<Row>, TuningError> {
        check_row(&row)?;
        match self.rows.binary_search_by(|r| r.key.cmp(&row.key)) {
            Ok(at) => Ok(Some(std::mem::replace(&mut self.rows[at], row))),
            Err(at) => {
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
            out.push_str(if i == 0 { "\n" } else { ",\n" });
            out.push_str(&format!(
                "    {{\"op\": {}, \"weights\": {}, \"shape\": {{{shape}}}, \"bucket\": {}, \"schedule\": {}, \"knobs\": {{{knobs}}}}}",
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
