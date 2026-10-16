//! Whole-head ownership over discontiguous source rows.
use crate::contracts::RowRange;
use crate::{Dim, Error, Geometry};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadSegment {
    pub id: String,
    pub heads: Dim,
    pub head_dim: Dim,
    pub ranges: Vec<RowRange>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadGatherOrder {
    #[default]
    Source,
    Segments(Vec<String>),
}
impl HeadGatherOrder {
    pub fn is_source(&self) -> bool {
        *self == Self::Source
    }
}

pub(crate) fn validate(
    segments: &[HeadSegment],
    order: &HeadGatherOrder,
    rows: u64,
    g: &Geometry,
) -> Result<(), Error> {
    if segments.is_empty() {
        return Err(Error("segmented ownership needs head segments".into()));
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut ranges = Vec::new();
    for segment in segments {
        if !crate::registry::valid_name(&segment.id)
            || !ids.insert(segment.id.as_str())
            || segment.ranges.is_empty()
        {
            return Err(Error(
                "head segments need unique ids and source ranges".into(),
            ));
        }
        let width = segment.head_dim.resolve(g)?;
        let mut total = 0u64;
        let mut previous = 0;
        for range in &segment.ranges {
            let (start, end) = crate::contract_validation::range(range, g)?;
            if start < previous || end > rows || (end - start) % width != 0 {
                return Err(Error(
                    "segment ranges must contain ordered whole heads".into(),
                ));
            }
            previous = end;
            total = total
                .checked_add(end - start)
                .ok_or_else(|| Error("segment extent overflows".into()))?;
            ranges.push((start, end));
        }
        if total != crate::validate::multiply(segment.heads.resolve(g)?, width)? {
            return Err(Error("segment rows differ from exact head geometry".into()));
        }
    }
    ranges.sort();
    let mut end = 0;
    for (start, next) in ranges {
        if start != end {
            return Err(Error(
                "head segments must cover source rows without overlap or gaps".into(),
            ));
        }
        end = next;
    }
    if end != rows {
        return Err(Error("head segments do not cover the source".into()));
    }
    if let HeadGatherOrder::Segments(order) = order {
        let ordered = order
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        if ordered != ids || order.len() != ids.len() {
            return Err(Error(
                "head gather order must name every segment exactly once".into(),
            ));
        }
    }
    Ok(())
}

pub(crate) fn select(
    segments: &[HeadSegment],
    order: &HeadGatherOrder,
    g: &Geometry,
    t: crate::derive::Topology,
) -> Result<Vec<std::ops::Range<u64>>, Error> {
    let tp = kernel_core::topology::TensorParallel::new(t.ranks as usize)
        .map_err(|e| Error(e.to_string()))?;
    let ordered: Vec<_> = match order {
        HeadGatherOrder::Source => segments.iter().collect(),
        HeadGatherOrder::Segments(ids) => ids
            .iter()
            .map(|id| {
                segments
                    .iter()
                    .find(|s| s.id == *id)
                    .ok_or_else(|| Error("missing head segment".into()))
            })
            .collect::<Result<_, _>>()?,
    };
    let mut selected = Vec::new();
    for segment in ordered {
        let heads = usize::try_from(segment.heads.resolve(g)?)
            .map_err(|_| Error("head count exceeds address space".into()))?;
        let owned = tp
            .kv_heads(heads)
            .and_then(|split| split.rank(t.rank as usize))
            .map_err(|e| Error(e.to_string()))?;
        let width = segment.head_dim.resolve(g)?;
        let first = owned.live.start as u64 * width;
        let last = owned.live.end as u64 * width;
        let mut logical = 0;
        for range in &segment.ranges {
            let (start, end) = crate::contract_validation::range(range, g)?;
            let length = end - start;
            let lo = first.max(logical);
            let hi = last.min(logical + length);
            if lo < hi {
                selected.push(start + lo - logical..start + hi - logical);
            }
            logical += length;
        }
    }
    if matches!(order, HeadGatherOrder::Source) {
        selected.sort_by_key(|range| range.start);
    }
    Ok(selected)
}
