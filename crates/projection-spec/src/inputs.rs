//! Activation inputs select logical segments independently of checkpoint tensors.
use crate::contracts::RowRange;
use crate::{Declaration, Dim, Error, Geometry, SliceAxis, SplitClass};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationInput {
    pub operand: String,
    pub binding: String,
    pub segments: Vec<ActivationSegment>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationSegment {
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Dim>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<Dim>,
    pub selection: ActivationSelection,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "select", rename_all = "snake_case", deny_unknown_fields)]
pub enum ActivationSelection {
    Rows { rows: RowRange },
    RankK { projection: String, operand: String },
}

impl ActivationSegment {
    /// Intersect the logical selection with this physical owner's domain.
    pub fn global_range(
        &self,
        declaration: &Declaration,
        geometry: &Geometry,
        topology: crate::derive::Topology,
    ) -> Result<std::ops::Range<u64>, Error> {
        let requested = self.requested_range(declaration, geometry, topology)?;
        match (&self.origin, &self.length) {
            (None, None) => Ok(requested),
            (Some(origin), Some(length)) => {
                let start = origin.resolve_offset(geometry)?;
                let end = start
                    .checked_add(length.resolve(geometry)?)
                    .ok_or_else(|| Error("activation source domain overflows".into()))?;
                let first = start.max(requested.start);
                let last = end.min(requested.end);
                Ok(if first < last {
                    first..last
                } else {
                    first..first
                })
            }
            _ => Err(Error(
                "activation source domains need both origin and length".into(),
            )),
        }
    }
    /// Return columns relative to the physical owner, including an empty
    /// selection when another owner contains this rank's logical K range.
    pub fn range(
        &self,
        declaration: &Declaration,
        geometry: &Geometry,
        topology: crate::derive::Topology,
    ) -> Result<std::ops::Range<u64>, Error> {
        let global = self.global_range(declaration, geometry, topology)?;
        if global.is_empty() {
            return Ok(0..0);
        }
        let origin = self
            .origin
            .as_ref()
            .map(|d| d.resolve_offset(geometry))
            .transpose()?
            .unwrap_or(0);
        Ok(global.start - origin..global.end - origin)
    }
    pub fn requested_range(
        &self,
        declaration: &Declaration,
        geometry: &Geometry,
        topology: crate::derive::Topology,
    ) -> Result<std::ops::Range<u64>, Error> {
        if topology.rank >= topology.ranks {
            return Err(Error("activation rank is outside topology".into()));
        }
        match &self.selection {
            ActivationSelection::Rows { rows } => {
                let (start, end) = crate::contract_validation::range(rows, geometry)?;
                Ok(start..end)
            }
            ActivationSelection::RankK {
                projection,
                operand,
            } => {
                let p = declaration
                    .projections
                    .iter()
                    .find(|p| p.id == *projection)
                    .ok_or_else(|| {
                        Error("activation segment names an unknown projection".into())
                    })?;
                let Some(index) = p.operand_index(operand) else {
                    return Err(Error("activation segment names an unknown operand".into()));
                };
                let k = p.shape[1].resolve(geometry)?;
                // A grouped dense input is every group's K side by side; a
                // rank reads its own groups' columns.
                if let crate::ProjectionKind::GroupedDense { .. } = p.kind {
                    let groups =
                        crate::derive::selection::select(p, geometry, topology, index as u32)?
                            .experts;
                    return Ok(u64::from(groups.start) * k..u64::from(groups.end) * k);
                }
                let tile: Option<&Dim> =
                    if let Some(split) = p.operand_splits.iter().find(|s| s.operand == *operand) {
                        (split.axis == SliceAxis::K).then_some(&split.tile)
                    } else {
                        match &p.split {
                            SplitClass::Row { tile }
                            | SplitClass::ExpertSharded {
                                axis: SliceAxis::K,
                                tile,
                            } => Some(tile),
                            _ => None,
                        }
                    };
                let tp = kernel_core::topology::TensorParallel::new(topology.ranks as usize)
                    .map_err(|e| Error(e.to_string()))?;
                if let Some(tile) = tile {
                    let split = tp
                        .padded_split(
                            usize::try_from(k)
                                .map_err(|_| Error("activation K exceeds address space".into()))?,
                            usize::try_from(tile.resolve(geometry)?).map_err(|_| {
                                Error("activation tile exceeds address space".into())
                            })?,
                        )
                        .map_err(|e| Error(e.to_string()))?;
                    let owned = split
                        .rank(topology.rank as usize)
                        .map_err(|e| Error(e.to_string()))?;
                    Ok(owned.live.start as u64..owned.live.end as u64)
                } else {
                    Ok(0..k)
                }
            }
        }
    }
}
