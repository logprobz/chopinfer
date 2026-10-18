//! Unit launches of a row op's case, for the invariance adapters that test
//! an entry's declared claims through the op SM120 prepares for it
//! (`tests/common/invariance.rs`).
//!
//! A unit is one row of a launch: the row's bytes of every live output, in
//! role order, then the bytes of each in-place state that the row's launch
//! alone writes ([`super::properties`]'s comparison). Row `r` of a launch
//! over images whose live inputs hold rows `order` of other images is that
//! row of the source, so an adapter seeds rows by choosing them from a
//! case's own images, which its builder made valid together (positions,
//! key lists, block tables).
//!
//! Every launch runs on poisoned outputs and poisoned state write maps and
//! is checked afterwards ([`Case::check_unit_writes`]): every element its
//! maps name written and, for BF16 and F32 elements, finite; every other
//! byte of an output untouched, and of a state kept but for its
//! read-modify-write ranges. Equal units of two launches that both left a
//! region unwritten would prove nothing.

use super::{OpCase, Tag};
use crate::conformance::{Expected, Images, Rows};
use crate::kit::Lookup;
use crate::{LabResult, Session};
use kernel_core::properties::Scope;
use std::collections::BTreeMap;
use std::ops::Range;

/// Each launched unit's bytes.
pub type Units = Vec<Vec<u8>>;

/// The roles of a row op's case: each live input and output with its row
/// bytes, and its in-place states.
#[derive(Clone, Debug, Default)]
pub struct RowRoles {
    pub inputs: Vec<(String, usize)>,
    pub outputs: Vec<(String, usize)>,
    pub states: Vec<String>,
}

impl OpCase<'_> {
    /// The case's images: every input's, ID input's and state's.
    pub fn images(&self) -> Images {
        self.case.images()
    }

    /// The rows the case's buffers hold.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The case's row roles under claims of `scope`, or why it is not a row
    /// op (an output of fixed rows other than a routed scope's metadata).
    pub fn row_roles(&self, scope: Scope) -> LabResult<RowRoles> {
        let mut roles = RowRoles::default();
        for leaf in &self.leaves {
            match (leaf.tag, leaf.rows) {
                (Tag::Input, Rows::Live) => roles.inputs.push((leaf.path.clone(), leaf.row_bytes)),
                (Tag::Output, Rows::Live) => {
                    roles.outputs.push((leaf.path.clone(), leaf.row_bytes))
                }
                (Tag::State, Rows::Live) => {
                    roles.inputs.push((leaf.path.clone(), leaf.row_bytes));
                    roles.outputs.push((leaf.path.clone(), leaf.row_bytes));
                }
                (Tag::State, Rows::Fixed(_)) => roles.states.push(leaf.path.clone()),
                (Tag::Output, Rows::Fixed(_))
                    if scope == Scope::RoutedOperation && leaf.route_metadata => {}
                (Tag::Output, Rows::Fixed(_)) => {
                    return Err(format!(
                        "output {} has fixed rows, so the case is not a row op",
                        leaf.path
                    ));
                }
                _ => {}
            }
        }
        if roles.inputs.is_empty() || roles.outputs.is_empty() && roles.states.is_empty() {
            return Err("the case has no input rows, or no output rows or states".into());
        }
        Ok(roles)
    }

    /// This case's images with each live input's first rows holding rows
    /// `order` of `source`, the images of a case over the same roles (this
    /// one's, or another op's of equal fixed images at any capacity); its
    /// rows past them keep this case's own. Every other input image is
    /// `source`'s, which must have this case's size; a state or zeroed
    /// image of another size keeps this case's own.
    pub fn select(&self, source: &Images, order: &[usize]) -> LabResult<Images> {
        if order.len() > self.capacity as usize {
            return Err(format!(
                "{} rows selected into a case of {}",
                order.len(),
                self.capacity
            ));
        }
        let own = self.images();
        let mut images = own.clone();
        let roles = self.row_roles(Scope::RoutedOperation).unwrap_or_default();
        for leaf in &self.leaves {
            let Ok(target) = own.bytes(&leaf.path) else {
                continue;
            };
            let from = source.bytes(&leaf.path)?;
            let bytes = match roles.inputs.iter().find(|(path, _)| *path == leaf.path) {
                Some(&(_, row)) => {
                    let mut bytes = target.to_vec();
                    for (to, &row_at) in order.iter().enumerate() {
                        let rows = from.len() / row;
                        if row_at >= rows {
                            return Err(format!(
                                "{}: row {row_at} of a source of {rows}",
                                leaf.path
                            ));
                        }
                        bytes[to * row..][..row].copy_from_slice(&from[row_at * row..][..row]);
                    }
                    bytes
                }
                None if from.len() == target.len() => from.to_vec(),
                // Zeroed semaphores and states sized by the capacity keep
                // this case's own (a state a launch reads must then come
                // from a case of the source's capacity).
                None if matches!(leaf.tag, Tag::State | Tag::Zeroed) => target.to_vec(),
                None => {
                    return Err(format!(
                        "{}: the source image holds {} bytes, this case's {}",
                        leaf.path,
                        from.len(),
                        target.len()
                    ));
                }
            };
            images = images.with(&leaf.path, bytes);
        }
        Ok(images)
    }

    /// The units of one launch of `rows` rows over `images` (this case's,
    /// as [`OpCase::select`] makes them), checked as the module says.
    pub fn units(
        &self,
        session: &Session,
        functions: &Lookup<'_>,
        images: &Images,
        rows: u32,
    ) -> LabResult<Units> {
        let roles = self.row_roles(self.info.guarantees.scope)?;
        let poisoned = self.case.poison_state_maps(images, rows)?;
        let results = self
            .case
            .execute(session, functions, &poisoned, rows, rows)?;
        self.case
            .check_unit_writes(&poisoned, rows, &results)
            .map_err(|error| format!("{rows} rows: {error}"))?;
        let mut units = Vec::with_capacity(rows as usize);
        for r in 0..rows as usize {
            let mut unit = Vec::new();
            for (path, row) in &roles.outputs {
                let bytes = results
                    .get(path)
                    .ok_or_else(|| format!("the launch gave no {path}"))?;
                unit.extend_from_slice(&bytes[r * row..][..*row]);
            }
            if !roles.states.is_empty() {
                let alone = self.select(images, &[r])?;
                for state in &roles.states {
                    let bytes = results
                        .get(state)
                        .ok_or_else(|| format!("the launch gave no {state}"))?;
                    for range in self.case.state_writes(state, &alone, 1)? {
                        unit.extend_from_slice(&bytes[range]);
                    }
                }
            }
            units.push(unit);
        }
        Ok(units)
    }

    /// Every output's and state's bytes after one launch of `rows` rows over
    /// `images`, checked as the module says: the images a chained launch
    /// (a prefill's next chunk) continues from.
    pub fn launch(
        &self,
        session: &Session,
        functions: &Lookup<'_>,
        images: &Images,
        rows: u32,
    ) -> LabResult<BTreeMap<String, Vec<u8>>> {
        let poisoned = self.case.poison_state_maps(images, rows)?;
        let results = self
            .case
            .execute(session, functions, &poisoned, rows, rows)?;
        self.case
            .check_unit_writes(&poisoned, rows, &results)
            .map_err(|error| format!("{rows} rows: {error}"))?;
        Ok(results)
    }

    /// Check a unit launch's outputs against this case's numerical or exact
    /// oracle. Use with [`Self::launch`], which checks writes and poison,
    /// when a test supplies new images or consumes an observed intermediate.
    pub fn judge_outputs(
        &self,
        images: &Images,
        rows: u32,
        outputs: &BTreeMap<String, Vec<u8>>,
    ) -> LabResult<serde_json::Value> {
        self.case.judge_outputs(images, rows, outputs)
    }

    /// The byte ranges of in-place state `name` a launch of `rows` rows
    /// over `images` writes or may rewrite.
    pub fn state_ranges(
        &self,
        name: &str,
        images: &Images,
        rows: u32,
    ) -> LabResult<Vec<Range<usize>>> {
        self.case.state_writes(name, images, rows)
    }

    /// The oracle's exact results at `rows` over `images`, by output: the
    /// outputs it states as bits (an exact selection's), not those it
    /// bounds.
    pub fn oracle_bits(&self, images: &Images, rows: u32) -> LabResult<BTreeMap<String, Vec<u8>>> {
        Ok(self
            .case
            .expected(images, rows)?
            .into_iter()
            .filter_map(|(name, expected)| match expected {
                Expected::Bits(bits) => Some((name, bits)),
                _ => None,
            })
            .collect())
    }

    /// The bytes a verify op's decode mapping compares ([`super::Decoded`]),
    /// one unit per decode launch: the verify step over the case's images
    /// at its capacity (`max` the claim's rows), then each decode launch
    /// alone, every launch checked as the module says. The first units are
    /// the verify step's, the second the decode launches'.
    pub fn decoded_units(
        &self,
        session: &Session,
        functions: &Lookup<'_>,
        max: u32,
    ) -> LabResult<(Units, Units)> {
        let Some(Ok(decoded)) = &self.decoded else {
            return Err("the case maps no verify item to decode launches".into());
        };
        let live = self.capacity;
        let execute = |images: &Images| self.launch(session, functions, images, live);
        let (_, verify, launches) = decoded.plan(&self.images(), live, max, &execute)?;
        let mut units = (Vec::new(), Vec::new());
        for launch in &launches {
            let decode = decoded.case.launch(session, functions, &launch.images, 1)?;
            let bytes = |results: &BTreeMap<String, Vec<u8>>,
                         (path, range): &(String, Range<usize>)| {
                results
                    .get(path)
                    .and_then(|bytes| bytes.get(range.clone()))
                    .map(<[u8]>::to_vec)
                    .ok_or_else(|| format!("{}: no bytes {range:?} of {path}", launch.label))
            };
            let (mut got, mut alone) = (Vec::new(), Vec::new());
            for compared in &launch.compared {
                got.extend(bytes(&verify, &compared.verify)?);
                alone.extend(bytes(&decode, &compared.decode)?);
            }
            units.0.push(got);
            units.1.push(alone);
        }
        Ok(units)
    }
}
