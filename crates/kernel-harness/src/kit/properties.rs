//! Checks of the exactness properties an op guarantees
//! (`kernel_core::properties`), each comparing output bits between launches
//! that the property says must agree.
//!
//! They hold for row ops, whose case binds rows of the live count
//! (`Rows::Live`) for inputs and outputs, shared fixed inputs and weights,
//! and scratch: each output row is then a function of its input rows, and
//! a launch of other rows is a launch over those rows moved to the front.
//! For an op over row `r` of `n` rows:
//!
//! - rows (`Rows::OneRow { max }`): every row `r` of a launch of `n` rows,
//!   `n` up to `max` and the capacity, has the bits of a one-row launch
//!   over row `r` alone (a verify step of row items against its rows as
//!   decode steps; items of several rows map to decode items through
//!   [`super::decode`]);
//! - batch (`Batch::Independent`): row `r` keeps its bits when every other
//!   row's inputs become copies of its own, for the first and last rows;
//! - chunk (`Chunk::Any` or `Granule(g)`): launches over rows `0..s` and
//!   `s..n` give the bits of one launch over `0..n`, at splits 1, `n / 2`
//!   and `n - 1` (multiples of `g` for a granule).
//!
//! In-place states ([`super::Spec::state`], such as a KV cache) are
//! compared where a row writes them: each check compares the state bytes
//! the compared row's launch alone writes (its write map and its
//! read-modify-write ranges at one live row over that row's inputs, as
//! [`super::Spec::state_updating`] gives them), which a launch whose state
//! depends on the row
//! count or on other rows changes. A case whose rows write overlapping
//! state bytes cannot use these checks. The evidence names the states and
//! the bytes compared.
//!
//! A launch that breaks a property changes some compared bit only where
//! the inputs make it: the inputs must differ from row to row, which the
//! kit's generators give. Ties and anchors depend on an op's own data (tied
//! values, anchor rows), so no row-generic check exists; an op claiming
//! them, and any property of an op whose case is not a row op, is reported
//! as not exercised, never as passed.
//!
//! Claims of the routed scope (`Scope::RoutedOperation`) are about the
//! token rows of the routed operation: outputs of fixed rows the case
//! marks route metadata ([`super::Spec::route_metadata`], which depends on
//! every routed row by design) are not compared, and the evidence names
//! them; any other output of fixed rows still makes the case no row op.

use super::{Check, Leaf, Tag};
use crate::LabResult;
use crate::conformance::{Images, Results, Rows};
use kernel_core::properties::{Batch, Chunk, Invariance, Property, Scope};
use serde_json::json;
use std::collections::BTreeSet;
use std::ops::Range;

/// The check's name in reports.
pub fn name(property: Property) -> String {
    format!("property {property:?}").to_lowercase()
}

/// Roles by path with their row bytes.
type RowRoles<'l> = Vec<(&'l str, usize)>;

/// A state's write map at a live count over images: the bytes a launch
/// writes.
pub(crate) type Written<'w> = dyn Fn(&str, &Images, u32) -> LabResult<Vec<Range<usize>>> + 'w;

/// The roles of a row op: the inputs and outputs a launch reads and writes
/// per row, its in-place states, and the route metadata a routed scope
/// leaves uncompared.
struct Layout<'l> {
    inputs: RowRoles<'l>,
    outputs: RowRoles<'l>,
    states: Vec<&'l str>,
    routed: Vec<&'l str>,
}

/// The row op's roles under claims of `scope`, or why the case is not a
/// row op.
fn rows(leaves: &[Leaf], scope: Scope) -> Result<Layout<'_>, String> {
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut states = Vec::new();
    let mut routed = Vec::new();
    for leaf in leaves {
        match (leaf.tag, leaf.rows) {
            (Tag::Input, Rows::Live) => inputs.push((leaf.path.as_str(), leaf.row_bytes)),
            (Tag::Output, Rows::Live) => outputs.push((leaf.path.as_str(), leaf.row_bytes)),
            (Tag::State, _) => states.push(leaf.path.as_str()),
            (Tag::Input | Tag::Weight | Tag::Zeroed | Tag::Scratch | Tag::Live, _) => {}
            (Tag::Output, Rows::Fixed(_))
                if scope == Scope::RoutedOperation && leaf.route_metadata =>
            {
                routed.push(leaf.path.as_str());
            }
            (Tag::Output, Rows::Fixed(_)) => {
                return Err(format!(
                    "output {} has fixed rows, so its rows are not the launch's rows",
                    leaf.path
                ));
            }
        }
    }
    if inputs.is_empty() || outputs.is_empty() && states.is_empty() {
        return Err("the case has no input rows, or no output rows or states".into());
    }
    Ok(Layout {
        inputs,
        outputs,
        states,
        routed,
    })
}

/// Compares the bytes `written` names of each state between two launches,
/// naming the first that differs; returns the bytes compared.
fn same_states(
    what: &str,
    states: &[&str],
    written: &dyn Fn(&str) -> LabResult<Vec<Range<usize>>>,
    left: &Results,
    right: &Results,
) -> LabResult<usize> {
    let mut compared = 0;
    for &state in states {
        let bytes = |results: &Results| {
            results
                .get(state)
                .cloned()
                .ok_or_else(|| format!("the launch gave no {state}"))
        };
        let (left, right) = (bytes(left)?, bytes(right)?);
        for range in written(state)? {
            compared += range.len();
            if let Some(at) = range.clone().find(|&at| left[at] != right[at]) {
                return Err(format!(
                    "{what}: state {state} byte {at} differs ({:#04x} and {:#04x})",
                    left[at], right[at]
                ));
            }
        }
    }
    Ok(compared)
}

/// `images` with each row input's rows `order` moved to the front, in
/// order; the rows past them keep their bytes.
fn front(images: &Images, inputs: &[(&str, usize)], order: &[usize]) -> LabResult<Images> {
    let mut moved = images.clone();
    for &(path, row) in inputs {
        let source = images.bytes(path)?;
        let mut bytes = source.to_vec();
        for (to, &from) in order.iter().enumerate() {
            bytes[to * row..][..row].copy_from_slice(&source[from * row..][..row]);
        }
        moved = moved.with(path, bytes);
    }
    Ok(moved)
}

/// Rows `range` of each output, concatenated per output.
fn output_rows(
    results: &Results,
    outputs: &[(&str, usize)],
    range: std::ops::Range<usize>,
) -> LabResult<Vec<Vec<u8>>> {
    outputs
        .iter()
        .map(|&(path, row)| {
            let bytes = results
                .get(path)
                .ok_or_else(|| format!("the launch gave no {path}"))?;
            Ok(bytes[range.start * row..range.end * row].to_vec())
        })
        .collect()
}

/// Compares two launches' rows, naming the first output that differs.
fn same(
    what: &str,
    outputs: &[(&str, usize)],
    left: &[Vec<u8>],
    right: &[Vec<u8>],
) -> LabResult<()> {
    for ((path, row), (left, right)) in outputs.iter().zip(left.iter().zip(right)) {
        if let Some(at) = left.iter().zip(right).position(|(a, b)| a != b) {
            return Err(format!(
                "{what}: {path} row {} byte {} differs ({:#04x} and {:#04x})",
                at / row,
                at % row,
                left[at],
                right[at]
            ));
        }
    }
    Ok(())
}

/// One check of each of `properties` at the claim `claims` makes of it,
/// over `execute`, which launches
/// the op once over images at a row count (with that live count) and gives
/// every output's bytes.
pub(crate) fn check(
    claims: Invariance,
    properties: &[Property],
    leaves: &[Leaf],
    images: &Images,
    capacity: u32,
    execute: &dyn Fn(&Images, u32) -> LabResult<Results>,
    written: &Written<'_>,
) -> Vec<Check> {
    let layout = rows(leaves, claims.scope);
    properties
        .iter()
        .map(|&property| {
            let name = name(property);
            let layout = match &layout {
                Ok(layout) => layout,
                Err(reason) => return Check::not_exercised(&name, reason),
            };
            let run = Runner {
                layout,
                images,
                execute,
                written,
            };
            let run = match property {
                Property::Rows => {
                    let max = match claims.rows {
                        kernel_core::properties::Rows::OneRow { max } => max,
                        kernel_core::properties::Rows::None => 0,
                    };
                    run.one_row(max.min(capacity))
                }
                Property::Batch if claims.batch == Batch::Independent => run.batch(capacity),
                Property::Chunk => run.chunk(claims.chunk, capacity),
                Property::Batch | Property::Ties | Property::Anchors => Err(Stop::Skip(NO_CHECK)),
            };
            match run {
                Ok(evidence) => Check::of(&name, Ok(evidence)),
                Err(Stop::Fail(error)) => Check::of(&name, Err(error)),
                Err(Stop::Skip(reason)) => Check::not_exercised(&name, reason),
            }
        })
        .collect()
}

/// [`check`] at each of `counts` as the launch's rows in place of the
/// capacity (counts below two are skipped), one check per property: it
/// fails at the first count that fails, passes with each count's evidence
/// when some count passed, and is otherwise not exercised for the first
/// count's reason. For ops whose launches change form with the row count
/// (a collective's protocols), where the capacity alone would check one
/// form.
#[allow(clippy::too_many_arguments)]
pub(crate) fn check_at(
    claims: Invariance,
    properties: &[Property],
    leaves: &[Leaf],
    images: &Images,
    counts: &[u32],
    execute: &dyn Fn(&Images, u32) -> LabResult<Results>,
    written: &Written<'_>,
) -> Vec<Check> {
    let runs = counts
        .iter()
        .filter(|&&count| count >= 2)
        .map(|&count| {
            (
                count,
                check(claims, properties, leaves, images, count, execute, written),
            )
        })
        .collect::<Vec<_>>();
    properties
        .iter()
        .enumerate()
        .map(|(at, &property)| {
            let name = name(property);
            let mut passed = serde_json::Map::new();
            let mut skipped = None;
            for (count, checks) in &runs {
                match &checks[at].outcome {
                    super::Outcome::Failed(error) => {
                        return Check::of(&name, Err(format!("at {count} rows: {error}")));
                    }
                    super::Outcome::Passed(evidence) => {
                        passed.insert(count.to_string(), evidence.clone());
                    }
                    super::Outcome::NotExercised(reason) => {
                        skipped.get_or_insert_with(|| reason.clone());
                    }
                }
            }
            if passed.is_empty() {
                let reason = skipped.unwrap_or_else(|| "no count of two rows or more".into());
                Check::not_exercised(&name, &reason)
            } else {
                Check::of(&name, Ok(json!({"by_rows": passed})))
            }
        })
        .collect()
}

/// Why a check stopped: it failed, or it cannot run for a reason.
enum Stop {
    Fail(String),
    Skip(&'static str),
}

type Run = Result<serde_json::Value, Stop>;

/// The reason a property no row-generic check covers is not exercised.
const NO_CHECK: &str =
    "the kit has no row-generic check of this property; the family's case must supply one";

/// One property check's launches over a row op's roles.
struct Runner<'r> {
    layout: &'r Layout<'r>,
    images: &'r Images,
    execute: &'r dyn Fn(&Images, u32) -> LabResult<Results>,
    written: &'r Written<'r>,
}

impl Runner<'_> {
    /// The images with each row input's rows `order` moved to the front.
    fn front(&self, order: &[usize]) -> Result<Images, Stop> {
        front(self.images, &self.layout.inputs, order).map_err(Stop::Fail)
    }

    fn execute(&self, images: &Images, rows: u32) -> Result<Results, Stop> {
        (self.execute)(images, rows).map_err(Stop::Fail)
    }

    /// Compares rows `left_rows` of `left` with `right_rows` of `right`.
    fn same_rows(
        &self,
        what: &str,
        (left, left_rows): (&Results, std::ops::Range<usize>),
        (right, right_rows): (&Results, std::ops::Range<usize>),
    ) -> Result<(), Stop> {
        let outputs = &self.layout.outputs;
        same(
            what,
            outputs,
            &output_rows(left, outputs, left_rows).map_err(Stop::Fail)?,
            &output_rows(right, outputs, right_rows).map_err(Stop::Fail)?,
        )
        .map_err(Stop::Fail)
    }

    /// Compares the state bytes a launch of `live` rows over `images`
    /// writes between `left` and `right`; returns the bytes compared.
    fn same_states(
        &self,
        what: &str,
        (images, live): (&Images, u32),
        left: &Results,
        right: &Results,
    ) -> Result<usize, Stop> {
        same_states(
            what,
            &self.layout.states,
            &|state| (self.written)(state, images, live),
            left,
            right,
        )
        .map_err(Stop::Fail)
    }

    fn one_row(&self, n: u32) -> Run {
        if n < 2 {
            return Err(Stop::Skip(
                "the op serves one row, so no launch has other rows",
            ));
        }
        let full = self.execute(self.images, n)?;
        let mut state_bytes = 0;
        for row in 0..n as usize {
            let what = format!("row {row} of {n} rows against the row alone");
            let images = self.front(&[row])?;
            let alone = self.execute(&images, 1)?;
            self.same_rows(&what, (&full, row..row + 1), (&alone, 0..1))?;
            state_bytes += self.same_states(&what, (&images, 1), &full, &alone)?;
        }
        Ok(json!({
            "rows": n,
            "compared_rows": "every row",
            "states": self.layout.states,
            "state_bytes": state_bytes,
            "routed_metadata_not_compared": self.layout.routed,
        }))
    }

    fn batch(&self, n: u32) -> Run {
        let n = n as usize;
        if n < 2 {
            return Err(Stop::Skip(
                "the op serves one row, so no launch has other rows",
            ));
        }
        let full = self.execute(self.images, n as u32)?;
        let rows = BTreeSet::from([0, n - 1]);
        let mut state_bytes = 0;
        for &row in &rows {
            // Every row becomes a copy of `row`, which a launch reading
            // other rows, even through a sum over them, sees.
            let what = format!("row {row} of {n} rows with every row a copy of it");
            let moved = self.execute(&self.front(&vec![row; n])?, n as u32)?;
            self.same_rows(&what, (&full, row..row + 1), (&moved, row..row + 1))?;
            // The state bytes the row writes alone, which every copy writes.
            let alone = self.front(&[row])?;
            state_bytes += self.same_states(&what, (&alone, 1), &full, &moved)?;
        }
        Ok(json!({
            "rows": n,
            "compared_rows": rows,
            "states": self.layout.states,
            "state_bytes": state_bytes,
            "routed_metadata_not_compared": self.layout.routed,
        }))
    }

    fn chunk(&self, claim: Chunk, n: u32) -> Run {
        let splits = match claim {
            Chunk::Any => BTreeSet::from([1, n / 2, n.saturating_sub(1)]),
            Chunk::Granule(granule) if granule > 0 => {
                (granule..n).step_by(granule as usize).collect()
            }
            Chunk::Granule(_) | Chunk::None => BTreeSet::new(),
        };
        let splits = splits
            .into_iter()
            .filter(|&split| split > 0 && split < n)
            .collect::<Vec<_>>();
        if splits.is_empty() {
            return Err(Stop::Skip(
                "no split of the capacity lies on the op's granule",
            ));
        }
        let full = self.execute(self.images, n)?;
        let mut state_bytes = 0;
        for &split in &splits {
            let what = format!("{n} rows against chunks split at {split}");
            let head = self.execute(self.images, split)?;
            let tail_images = self.front(&(split as usize..n as usize).collect::<Vec<_>>())?;
            let tail = self.execute(&tail_images, n - split)?;
            let (split_at, rest) = (split as usize, (n - split) as usize);
            self.same_rows(&what, (&full, 0..split_at), (&head, 0..split_at))?;
            self.same_rows(&what, (&full, split_at..n as usize), (&tail, 0..rest))?;
            state_bytes += self.same_states(&what, (self.images, split), &full, &head)?;
            state_bytes += self.same_states(&what, (&tail_images, n - split), &full, &tail)?;
        }
        Ok(json!({
            "rows": n,
            "splits": splits,
            "states": self.layout.states,
            "state_bytes": state_bytes,
            "routed_metadata_not_compared": self.layout.routed,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::super::Outcome;
    use super::*;

    const WIDTH: usize = 2;

    fn leaves() -> Vec<Leaf> {
        let leaf = |path: &str, rows, tag| Leaf {
            path: path.into(),
            rows,
            row_bytes: WIDTH,
            tag,
            route_metadata: false,
        };
        vec![
            leaf("x", Rows::Live, Tag::Input),
            leaf("weight", Rows::Fixed(1), Tag::Weight),
            leaf("y", Rows::Live, Tag::Output),
            leaf("live_rows", Rows::Fixed(1), Tag::Live),
        ]
    }

    const CAPACITY: u32 = 8;

    fn images() -> Images {
        let x = (0..CAPACITY as usize * WIDTH)
            .map(|byte| (byte * 7 + 3) as u8)
            .collect();
        Images::default().with("x", x).with("weight", vec![5, 9])
    }

    /// A host model of a row op: each output byte from its row's input
    /// bytes, the launch's row count and every live row, as `mix` says.
    fn op(mix: impl Fn(u8, u32, u8) -> u8) -> impl Fn(&Images, u32) -> LabResult<Results> {
        move |images: &Images, rows: u32| {
            let x = images.bytes("x")?;
            let sum = x[..rows as usize * WIDTH]
                .iter()
                .fold(0u8, |sum, &byte| sum.wrapping_add(byte));
            let mut y = vec![0xff; CAPACITY as usize * WIDTH];
            for (at, byte) in y.iter_mut().take(rows as usize * WIDTH).enumerate() {
                *byte = mix(x[at], rows, sum);
            }
            Ok(Results::from([("y".to_owned(), y)]))
        }
    }

    fn outcomes(
        claims: Invariance,
        execute: &dyn Fn(&Images, u32) -> LabResult<Results>,
    ) -> Vec<(String, bool)> {
        check(
            claims,
            &claims.claims(),
            &leaves(),
            &images(),
            CAPACITY,
            execute,
            &written,
        )
        .into_iter()
        .map(|check| {
            let passed = match check.outcome {
                Outcome::Passed(_) => true,
                Outcome::Failed(_) => false,
                Outcome::NotExercised(reason) => panic!("{}: {reason}", check.name),
            };
            (check.name, passed)
        })
        .collect()
    }

    fn passes(outcomes: &[(String, bool)], property: Property) -> bool {
        outcomes
            .iter()
            .find(|(name, _)| *name == super::name(property))
            .expect("checked")
            .1
    }

    /// Every check passes a row-local op, and each fails the launch that
    /// breaks its property.
    #[test]
    fn each_property_check_refuses_the_launch_that_breaks_it() {
        let claims = Invariance::any_rows(CAPACITY);
        let exact = outcomes(claims, &op(|x, _, _| x.wrapping_mul(3)));
        assert!(exact.iter().all(|(_, passed)| *passed), "{exact:?}");
        // Bits that follow the launch's row count break rows and chunks,
        // not batch independence.
        let counted = outcomes(claims, &op(|x, rows, _| x.wrapping_add(rows as u8)));
        assert!(!passes(&counted, Property::Rows));
        assert!(!passes(&counted, Property::Chunk));
        assert!(passes(&counted, Property::Batch));
        // Bits that read other rows break all three.
        let mixed = outcomes(claims, &op(|x, _, sum| x ^ sum));
        assert!(!passes(&mixed, Property::Rows));
        assert!(!passes(&mixed, Property::Chunk));
        assert!(!passes(&mixed, Property::Batch));
    }

    /// The rows check compares every row, not samples: a launch whose bits
    /// follow the row count at row 1 alone (which the first, middle and
    /// last rows of eight miss) fails it.
    #[test]
    fn the_rows_check_compares_every_row() {
        let claims = Invariance::any_rows(CAPACITY);
        let row_one = images().bytes("x").unwrap()[WIDTH];
        let counted = outcomes(
            claims,
            &op(move |x, rows, _| if x == row_one && rows > 1 { x ^ 1 } else { x }),
        );
        assert!(!passes(&counted, Property::Rows), "{counted:?}");
    }

    /// Routed-scope claims compare the token rows and leave the outputs
    /// the case marks route metadata uncompared, naming them; the same
    /// claims of an op's own rows find no row op, and neither do
    /// routed-scope claims over an unmarked output of fixed rows (token
    /// values of a fixed capacity whose bits follow the row count: review
    /// R1 found it passed all three checks when every fixed output was
    /// taken for metadata).
    #[test]
    fn a_routed_scope_leaves_route_metadata_uncompared() {
        let mut leaves = leaves();
        leaves.push(Leaf {
            path: "meta".into(),
            rows: Rows::Fixed(1),
            row_bytes: WIDTH,
            tag: Tag::Output,
            route_metadata: true,
        });
        // Metadata from every row and the row count, as route offsets.
        let execute = |images: &Images, rows: u32| {
            let mut results = op(|x, _, _| x.wrapping_mul(3))(images, rows)?;
            let sum = images.bytes("x")?[..rows as usize * WIDTH]
                .iter()
                .fold(rows as u8, |sum, &byte| sum.wrapping_add(byte));
            results.insert("meta".into(), vec![sum; WIDTH]);
            Ok(results)
        };
        let routed = Invariance {
            scope: Scope::RoutedOperation,
            ..Invariance::any_rows(CAPACITY)
        };
        let checks = check(
            routed,
            &routed.claims(),
            &leaves,
            &images(),
            CAPACITY,
            &execute,
            &written,
        );
        assert_eq!(checks.len(), 3);
        for check in &checks {
            let Outcome::Passed(evidence) = &check.outcome else {
                panic!("{check:?}");
            };
            assert_eq!(evidence["routed_metadata_not_compared"], json!(["meta"]));
        }
        let own = Invariance::any_rows(CAPACITY);
        let checks = check(
            own,
            &own.claims(),
            &leaves,
            &images(),
            CAPACITY,
            &execute,
            &written,
        );
        assert!(
            checks
                .iter()
                .all(|check| matches!(&check.outcome, Outcome::NotExercised(reason) if reason.contains("fixed rows"))),
            "{checks:?}"
        );
        leaves.last_mut().expect("meta").route_metadata = false;
        let checks = check(
            routed,
            &routed.claims(),
            &leaves,
            &images(),
            CAPACITY,
            &execute,
            &written,
        );
        assert!(
            checks
                .iter()
                .all(|check| matches!(&check.outcome, Outcome::NotExercised(reason) if reason.contains("output meta has fixed rows"))),
            "{checks:?}"
        );
    }

    /// Checks at each transition count catch a launch wrong only at one
    /// row count (17 rows here, where a four-rank push changes protocol),
    /// which the checks at the capacity alone pass: rows and chunk fail at
    /// 17 rows (review R1's mutant).
    #[test]
    fn checks_at_each_count_catch_a_form_of_one_count() {
        const ROWS: usize = 32;
        let images = Images::default().with(
            "x",
            (0..ROWS * WIDTH).map(|byte| (byte * 7 + 3) as u8).collect(),
        );
        let execute = |images: &Images, rows: u32| {
            let x = images.bytes("x")?;
            let mut y = vec![0xff; ROWS * WIDTH];
            for (at, byte) in y.iter_mut().take(rows as usize * WIDTH).enumerate() {
                *byte = if rows == 17 { x[at] ^ 1 } else { x[at] };
            }
            Ok(Results::from([("y".to_owned(), y)]))
        };
        let claims = Invariance::any_rows(ROWS as u32);
        let at_capacity = check(
            claims,
            &claims.claims(),
            &leaves(),
            &images,
            ROWS as u32,
            &execute,
            &written,
        );
        assert!(
            at_capacity
                .iter()
                .all(|check| matches!(check.outcome, Outcome::Passed(_))),
            "{at_capacity:?}"
        );
        let at_counts = check_at(
            claims,
            &claims.claims(),
            &leaves(),
            &images,
            &[1, 2, 3, 16, 17, 32],
            &execute,
            &written,
        );
        assert_eq!(at_counts.len(), 3);
        for check in &at_counts {
            // Bits that follow the row count alone keep batch independence.
            if check.name == name(Property::Batch) {
                assert!(matches!(check.outcome, Outcome::Passed(_)), "{check:?}");
                continue;
            }
            let Outcome::Failed(error) = &check.outcome else {
                panic!("{check:?}");
            };
            assert!(error.starts_with("at 17 rows"), "{error}");
        }
    }

    /// The state byte a row writes: its first input byte over 14, which is
    /// the row's index for the case's images.
    fn target(images: &Images, row: usize) -> LabResult<usize> {
        Ok(usize::from(images.bytes("x")?[row * WIDTH]) / 14)
    }

    fn written(state: &str, images: &Images, live: u32) -> LabResult<Vec<Range<usize>>> {
        assert_eq!(state, "s");
        (0..live as usize)
            .map(|row| target(images, row).map(|at| at..at + 1))
            .collect()
    }

    /// A row op with exact outputs whose state byte at each row's target is
    /// `mix` of its input byte, the row count and the sum of every row.
    fn stateful(mix: impl Fn(u8, u32, u8) -> u8) -> impl Fn(&Images, u32) -> LabResult<Results> {
        move |images: &Images, rows: u32| {
            let mut results = op(|x, _, _| x)(images, rows)?;
            let x = images.bytes("x")?;
            let sum = x[..rows as usize * WIDTH]
                .iter()
                .fold(0u8, |sum, &byte| sum.wrapping_add(byte));
            let mut state = vec![0x5a; CAPACITY as usize];
            for row in 0..rows as usize {
                state[target(images, row)?] = mix(x[row * WIDTH], rows, sum);
            }
            results.insert("s".into(), state);
            Ok(results)
        }
    }

    /// A launch with exact row outputs whose state follows the row count or
    /// other rows fails the checks, with or without row outputs.
    #[test]
    fn property_checks_compare_the_state_each_row_writes() {
        let claims = Invariance::any_rows(CAPACITY);
        let state = Leaf {
            path: "s".into(),
            rows: Rows::Fixed(1),
            row_bytes: CAPACITY as usize,
            tag: Tag::State,
            route_metadata: false,
        };
        let with_state = leaves()
            .into_iter()
            .chain([state.clone()])
            .collect::<Vec<_>>();
        let state_only = leaves()
            .into_iter()
            .filter(|leaf| leaf.tag != Tag::Output)
            .chain([state])
            .collect::<Vec<_>>();
        for leaves in [&with_state, &state_only] {
            let run = |execute: &dyn Fn(&Images, u32) -> LabResult<Results>| {
                check(
                    claims,
                    &claims.claims(),
                    leaves,
                    &images(),
                    CAPACITY,
                    execute,
                    &written,
                )
            };
            let exact = run(&stateful(|x, _, _| x.wrapping_mul(5)));
            assert!(
                exact.iter().all(|check| matches!(&check.outcome,
                    Outcome::Passed(evidence) if evidence["state_bytes"].as_u64() > Some(0))),
                "{exact:?}"
            );
            let mixed = run(&stateful(|x, _, sum| x ^ sum));
            assert!(
                mixed
                    .iter()
                    .all(|check| matches!(&check.outcome, Outcome::Failed(error) if error.contains("state s"))),
                "{mixed:?}"
            );
            let counted = run(&stateful(|x, rows, _| x.wrapping_add(rows as u8)));
            for property in [Property::Rows, Property::Chunk] {
                let check = counted
                    .iter()
                    .find(|check| check.name == name(property))
                    .unwrap();
                assert!(matches!(check.outcome, Outcome::Failed(_)), "{check:?}");
            }
        }
    }

    /// Unchecked claims are reported, never passed.
    #[test]
    fn ties_anchors_and_non_row_cases_are_not_exercised() {
        let claims = Invariance::NONE
            .with_ties(kernel_core::properties::Ties::LowerIndex)
            .with_anchors(kernel_core::properties::Anchors::Spare);
        let execute = op(|x, _, _| x);
        let checks = check(
            claims,
            &claims.claims(),
            &leaves(),
            &images(),
            CAPACITY,
            &execute,
            &written,
        );
        assert_eq!(checks.len(), 2);
        assert!(
            checks
                .iter()
                .all(|check| matches!(check.outcome, Outcome::NotExercised(_)))
        );
        let mut fixed = leaves();
        fixed[2].rows = Rows::Fixed(4);
        let checks = check(
            Invariance::any_rows(CAPACITY),
            &Invariance::any_rows(CAPACITY).claims(),
            &fixed,
            &images(),
            CAPACITY,
            &execute,
            &written,
        );
        assert_eq!(checks.len(), 3);
        assert!(
            checks
                .iter()
                .all(|check| matches!(&check.outcome, Outcome::NotExercised(reason) if reason.contains("fixed rows")))
        );
    }
}
