//! Drafter membership and published-port reuse. Both are optional additions
//! to schema v1: a declaration without them keeps its canonical identity.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::bind::SourceOwner;
use crate::{Consumer, Declaration, Error, Geometry, Projection, SideTensor};

/// An optional drafter: the layer stems and stemless projections it owns,
/// and the projections whose prepared operations it records all or none.
/// Fusion groups and side tensors belong to it through their members,
/// source stems or consumers.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Drafter {
    pub id: String,
    pub stems: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub projections: Vec<String>,
    pub operations: Vec<String>,
}

impl Drafter {
    /// Whether `projection` belongs to this drafter.
    pub fn owns(&self, projection: &Projection) -> bool {
        projection
            .stem()
            .is_some_and(|stem| self.stems.iter().any(|s| s == stem))
            || self.projections.contains(&projection.id)
    }
}

impl Declaration {
    /// The drafter that owns projection `id`, if any.
    pub fn drafter_of(&self, id: &str) -> Option<&Drafter> {
        let projection = self.projections.iter().find(|p| p.id == id)?;
        self.drafters.iter().find(|d| d.owns(projection))
    }

    /// The projection whose published port serves `id`: its declared
    /// `publishes_as`, else itself.
    pub fn published<'a>(&'a self, id: &'a str) -> &'a str {
        self.projections
            .iter()
            .find(|p| p.id == id)
            .and_then(|p| p.publishes_as.as_deref())
            .unwrap_or(id)
    }

    /// The drafter that owns fusion group `id`: the owner of its members.
    pub fn drafter_of_group(&self, id: &str) -> Option<&Drafter> {
        let group = self.fusion_groups.iter().find(|g| g.id == id)?;
        self.drafter_of(group.members.first()?)
    }

    /// The drafter that owns a side tensor: it reads one of the drafter's
    /// stems or a drafter projection or fusion group consumes it.
    pub fn drafter_of_side(&self, side: &SideTensor) -> Option<&Drafter> {
        let consumer = match &side.consumer {
            Consumer::Declared(id) => self.drafter_of(id).or_else(|| self.drafter_of_group(id)),
            Consumer::Kernel { .. } => None,
        };
        consumer.or_else(|| {
            let stem = side.source.stem.as_ref()?;
            self.drafters.iter().find(|d| d.stems.contains(stem))
        })
    }

    /// The checkpoint tensors every drafter owns: its projections' sources
    /// and its side tensors.
    pub fn drafter_sources(&self, geometry: &Geometry) -> Result<BTreeSet<String>, Error> {
        Ok(self
            .source_positions(geometry)?
            .into_iter()
            .filter(|(_, position)| match position.owner {
                SourceOwner::Projection(index) => {
                    self.drafter_of(&self.projections[index].id).is_some()
                }
                SourceOwner::Side(index) => {
                    self.drafter_of_side(&self.side_tensors[index]).is_some()
                }
            })
            .map(|(name, _)| name)
            .collect())
    }

    /// The materialized stem index a projection reads at executed `layer`:
    /// its stem's execution alias, else the same index.
    pub fn source_layer(&self, geometry: &Geometry, id: &str, layer: u64) -> Result<u64, Error> {
        let projection = self
            .projections
            .iter()
            .find(|p| p.id == id)
            .ok_or_else(|| Error(format!("unknown projection `{id}`")))?;
        match projection
            .stem()
            .and_then(|stem| self.layer_stems.iter().find(|s| s.id == stem))
        {
            Some(stem) => stem.source_index(geometry, layer),
            None => Ok(layer),
        }
    }

    /// The projection executed at `layer` whose published port is
    /// `published`: that projection at its own layers, or one that publishes
    /// as it at another stem's layers.
    pub fn executed_at(
        &self,
        geometry: &Geometry,
        layer: u64,
        published: &str,
    ) -> Result<&Projection, Error> {
        for p in &self.projections {
            if self.published(&p.id) != published {
                continue;
            }
            let Some(stem) = p
                .stem()
                .and_then(|stem| self.layer_stems.iter().find(|s| s.id == stem))
            else {
                continue;
            };
            if stem.executed(geometry)?.contains(&layer) {
                return Ok(p);
            }
        }
        Err(Error(format!(
            "no projection publishes as `{published}` at layer {layer}"
        )))
    }

    /// Remove drafter `id` with every member: its stems, their projections,
    /// its stemless projections, fusion groups of its projections, and side
    /// tensors read through its stems or consumed by its members.
    pub fn omit_drafter(&mut self, id: &str) -> Result<(), Error> {
        if !self.drafters.iter().any(|d| d.id == id) {
            return Err(Error(format!("unknown drafter `{id}`")));
        }
        let owned = |owner: Option<&Drafter>| owner.is_some_and(|d| d.id == id);
        let projections: BTreeSet<String> = self
            .projections
            .iter()
            .filter(|p| owned(self.drafter_of(&p.id)))
            .map(|p| p.id.clone())
            .collect();
        let groups: BTreeSet<String> = self
            .fusion_groups
            .iter()
            .filter(|g| owned(self.drafter_of_group(&g.id)))
            .map(|g| g.id.clone())
            .collect();
        let sides: BTreeSet<String> = self
            .side_tensors
            .iter()
            .filter(|s| owned(self.drafter_of_side(s)))
            .map(|s| s.id.clone())
            .collect();
        let index = self.drafters.iter().position(|d| d.id == id).unwrap();
        let drafter = self.drafters.remove(index);
        self.projections.retain(|p| !projections.contains(&p.id));
        self.fusion_groups.retain(|g| !groups.contains(&g.id));
        self.side_tensors.retain(|s| !sides.contains(&s.id));
        self.layer_stems.retain(|s| !drafter.stems.contains(&s.id));
        Ok(())
    }

    pub(crate) fn validate_drafters(&self) -> Result<(), Error> {
        let find = |id: &str| self.projections.iter().find(|p| p.id == id);
        for p in &self.projections {
            let Some(target) = p.publishes_as.as_deref() else {
                continue;
            };
            let refuse = |reason: &str| {
                Err(Error(format!(
                    "projection `{}` cannot publish as `{target}`: {reason}",
                    p.id
                )))
            };
            let Some(t) = find(target).filter(|t| t.id != p.id) else {
                return refuse("no other declared projection");
            };
            if t.publishes_as.is_some() {
                return refuse("the target publishes as another projection");
            }
            if p.kind != t.kind
                || p.shape != t.shape
                || p.operands != t.operands
                || p.parts != t.parts
                || p.split != t.split
                || p.epilogues != t.epilogues
                || p.numerical_contract != t.numerical_contract
                || p.format != t.format
                || p.uses != t.uses
            {
                return refuse(
                    "kind, geometry, operands, split, epilogues, numerics or uses differ",
                );
            }
            // Per-operand padding, activation K ownership, source transforms
            // and lookup head ownership also fix the physical layout the
            // reused entries read.
            if p.operand_splits != t.operand_splits
                || p.inputs != t.inputs
                || p.source_transforms != t.source_transforms
                || p.lookup.as_ref().map(lookup_layout) != t.lookup.as_ref().map(lookup_layout)
            {
                return refuse("its physical layout differs");
            }
            // The projection keeps its own format admission, which may be
            // narrower than the target's, never wider or different.
            let compatible = match (&p.composition, &t.composition) {
                (None, None) => true,
                (Some(own), Some(theirs)) => {
                    own.pre == theirs.pre
                        && own.post == theirs.post
                        && own.implementation == theirs.implementation
                        && own.numerical_contracts.iter().all(|(format, recipe)| {
                            theirs.numerical_contracts.get(format) == Some(recipe)
                                && own.implementation_for(*format)
                                    == theirs.implementation_for(*format)
                        })
                }
                _ => false,
            };
            if !compatible {
                return refuse("its composition admits a recipe the target does not publish");
            }
        }
        let mut ids = BTreeSet::new();
        let mut stems = BTreeSet::new();
        let mut members = BTreeSet::new();
        for drafter in &self.drafters {
            if !crate::registry::valid_name(&drafter.id) || !ids.insert(&drafter.id) {
                return Err(Error(format!(
                    "invalid or duplicate drafter `{}`",
                    drafter.id
                )));
            }
            let refuse = |reason: String| Err(Error(format!("drafter `{}` {reason}", drafter.id)));
            for stem in &drafter.stems {
                if !self.layer_stems.iter().any(|s| s.id == *stem) || !stems.insert(stem) {
                    return refuse(format!("names an unknown or claimed stem `{stem}`"));
                }
            }
            for id in &drafter.projections {
                let Some(p) = find(id) else {
                    return refuse(format!("names an unknown projection `{id}`"));
                };
                if p.stem()
                    .is_some_and(|stem| drafter.stems.iter().any(|s| s == stem))
                {
                    return refuse(format!("lists stem member `{id}` again"));
                }
            }
            for p in self.projections.iter().filter(|p| drafter.owns(p)) {
                if !members.insert(&p.id) {
                    return refuse(format!("shares projection `{}`", p.id));
                }
            }
            let mut operations = BTreeSet::new();
            for id in &drafter.operations {
                if !find(id).is_some_and(|p| drafter.owns(p)) || !operations.insert(id) {
                    return refuse(format!("records non-member or repeated operation `{id}`"));
                }
            }
        }
        for group in &self.fusion_groups {
            let owners: BTreeSet<_> = group
                .members
                .iter()
                .map(|m| self.drafter_of(m).map(|d| &d.id))
                .collect();
            if owners.len() > 1 {
                return Err(Error(format!(
                    "fusion `{}` mixes drafter and other members",
                    group.id
                )));
            }
        }
        Ok(())
    }
}

/// A lookup's layout without its checkpoint shard sources, which an alias
/// names for itself.
fn lookup_layout(lookup: &crate::contracts::LookupAssembly) -> serde_json::Value {
    let mut value = serde_json::to_value(lookup).expect("lookup serialization is infallible");
    match &mut value["shards"] {
        serde_json::Value::Array(shards) => {
            for shard in shards {
                if let Some(shard) = shard.as_object_mut() {
                    shard.remove("source");
                }
            }
        }
        serde_json::Value::Object(template) => {
            template.remove("source");
        }
        _ => {}
    }
    value
}
