//! Axis placement for arbitrary-rank side tensors. Conversion follows placement.
use crate::{DType, Dim, Error, Geometry, SideTensor, SplitClass};
use serde::{Deserialize, Serialize};
use std::ops::Range;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidePlacement {
    #[serde(default)]
    pub axis: usize,
    pub split: SplitClass,
}
impl Default for SidePlacement {
    fn default() -> Self {
        Self {
            axis: 0,
            split: SplitClass::Replicate {},
        }
    }
}
impl SidePlacement {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SideSelection {
    pub axis: Option<usize>,
    pub ranges: Vec<Range<u64>>,
    pub source_shape: Vec<u64>,
    pub shape: Vec<u64>,
}

impl SideTensor {
    pub fn selection(
        &self,
        g: &Geometry,
        t: crate::derive::Topology,
    ) -> Result<SideSelection, Error> {
        let source_shape = crate::contract_validation::shape(&self.shape, g)?;
        let tp = kernel_core::topology::TensorParallel::new(t.ranks as usize)
            .map_err(|e| Error(e.to_string()))?;
        if t.rank >= t.ranks {
            return Err(Error("side rank is outside topology".into()));
        }
        if source_shape.is_empty() {
            if !self.placement.is_default() {
                return Err(Error("scalar sides must replicate".into()));
            }
            return Ok(SideSelection {
                axis: None,
                ranges: std::iter::once(0..1).collect(),
                shape: source_shape.clone(),
                source_shape,
            });
        }
        let axis = self.placement.axis;
        let rows = *source_shape
            .get(axis)
            .ok_or_else(|| Error("side split axis is out of bounds".into()))?;
        let tile = |tile: &Dim| tile.resolve(g);
        let mut physical = rows;
        let ranges = match &self.placement.split {
            SplitClass::Replicate {} => std::iter::once(0..rows).collect(),
            SplitClass::HeadGrouped {
                heads,
                head_dim,
                tile: padding,
            } => {
                let heads = usize::try_from(heads.resolve(g)?)
                    .map_err(|_| Error("side head count exceeds address space".into()))?;
                let owned = tp
                    .kv_heads(heads)
                    .and_then(|s| s.rank(t.rank as usize))
                    .map_err(|e| Error(e.to_string()))?;
                let width = head_dim.resolve(g)?;
                let range = owned.live.start as u64 * width..owned.live.end as u64 * width;
                physical = super::derive::selection::pad(range.end - range.start, tile(padding)?)?;
                std::iter::once(range).collect()
            }
            SplitClass::SegmentedHeads {
                segments,
                order,
                tile: padding,
            } => {
                let ranges = crate::segments::select(segments, order, g, t)?;
                physical = super::derive::selection::pad(
                    ranges.iter().map(|r| r.end - r.start).sum(),
                    tile(padding)?,
                )?;
                ranges
            }
            SplitClass::Column { tile: padding }
            | SplitClass::Row { tile: padding }
            | SplitClass::VocabSharded { tile: padding }
            | SplitClass::ExpertSharded { tile: padding, .. } => {
                let split = tp
                    .padded_split(
                        usize::try_from(rows)
                            .map_err(|_| Error("side dimension exceeds address space".into()))?,
                        usize::try_from(tile(padding)?)
                            .map_err(|_| Error("side tile exceeds address space".into()))?,
                    )
                    .map_err(|e| Error(e.to_string()))?;
                let owned = split
                    .rank(t.rank as usize)
                    .map_err(|e| Error(e.to_string()))?;
                physical = owned.padded_len() as u64;
                std::iter::once(owned.live.start as u64..owned.live.end as u64).collect()
            }
            SplitClass::ExpertOwned { tile: padding } => {
                let owned = tp
                    .split(
                        usize::try_from(rows)
                            .map_err(|_| Error("side dimension exceeds address space".into()))?,
                    )
                    .and_then(|s| s.rank(t.rank as usize))
                    .map_err(|e| Error(e.to_string()))?;
                physical = super::derive::selection::pad(owned.live_len() as u64, tile(padding)?)?;
                std::iter::once(owned.live.start as u64..owned.live.end as u64).collect()
            }
        };
        let mut shape = source_shape.clone();
        shape[axis] = physical;
        Ok(SideSelection {
            axis: Some(axis),
            ranges,
            source_shape,
            shape,
        })
    }

    pub(crate) fn validate_placement(&self, g: &Geometry) -> Result<(), Error> {
        let shape = crate::contract_validation::shape(&self.shape, g)?;
        if shape.is_empty() {
            if self.placement.is_default() {
                return Ok(());
            }
            return Err(Error("scalar sides must replicate".into()));
        }
        let rows = *shape
            .get(self.placement.axis)
            .ok_or_else(|| Error("side split axis is out of bounds".into()))?;
        match &self.placement.split {
            SplitClass::Replicate {} => {}
            SplitClass::HeadGrouped {
                heads,
                head_dim,
                tile,
            } => {
                tile.resolve(g)?;
                if crate::validate::multiply(heads.resolve(g)?, head_dim.resolve(g)?)? != rows {
                    return Err(Error("side head geometry differs from split axis".into()));
                }
            }
            SplitClass::SegmentedHeads {
                segments,
                order,
                tile,
            } => {
                tile.resolve(g)?;
                crate::segments::validate(segments, order, rows, g)?;
            }
            SplitClass::Column { tile }
            | SplitClass::Row { tile }
            | SplitClass::VocabSharded { tile }
            | SplitClass::ExpertOwned { tile }
            | SplitClass::ExpertSharded { tile, .. } => {
                tile.resolve(g)?;
            }
        }
        Ok(())
    }
}

impl SideSelection {
    /// Gather and zero-pad one physical axis without changing element bits.
    pub fn repack(&self, source: &[u8], dtype: DType) -> Result<Vec<u8>, Error> {
        let bits = kernel_core::precision::dtype_bits(dtype);
        if bits == 0 || bits % 8 != 0 {
            return Err(Error(
                "side placement needs byte-addressable elements".into(),
            ));
        }
        let width = bits / 8;
        let elements = self
            .source_shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .ok_or_else(|| Error("side source extent overflows".into()))?;
        let bytes = elements
            .checked_mul(width)
            .ok_or_else(|| Error("side source bytes overflow".into()))?;
        if source.len() as u64 != bytes {
            return Err(Error("side source bytes differ from declared shape".into()));
        }
        if self.shape.len() != self.source_shape.len()
            || self.source_shape.contains(&0)
            || self.shape.contains(&0)
        {
            return Err(Error("side selection has invalid shape geometry".into()));
        }
        let Some(axis) = self.axis else {
            return Ok(source.to_vec());
        };
        if axis >= self.source_shape.len()
            || self
                .shape
                .iter()
                .zip(&self.source_shape)
                .enumerate()
                .any(|(index, (a, b))| index != axis && a != b)
        {
            return Err(Error("side selection changes an unsplit axis".into()));
        }
        let inner = self.source_shape[axis + 1..].iter().product::<u64>();
        let outer = self.source_shape[..axis].iter().product::<u64>();
        let row_bytes = inner
            .checked_mul(width)
            .ok_or_else(|| Error("side row extent overflows".into()))?;
        let output = outer
            .checked_mul(self.shape[axis])
            .and_then(|n| n.checked_mul(row_bytes))
            .filter(|n| *n <= kernel_core::precision::MAX_TENSOR_BYTES)
            .ok_or_else(|| Error("side output extent exceeds supported bound".into()))?;
        let mut result = vec![
            0;
            usize::try_from(output)
                .map_err(|_| Error("side output exceeds address space".into()))?
        ];
        for outer in 0..outer {
            let mut cursor = 0;
            for range in &self.ranges {
                if range.end > self.source_shape[axis] || range.start > range.end {
                    return Err(Error("side selection exceeds source axis".into()));
                }
                for row in range.clone() {
                    if cursor >= self.shape[axis] {
                        return Err(Error("side rows exceed padded output".into()));
                    }
                    let start = ((outer * self.source_shape[axis] + row) * row_bytes) as usize;
                    let target = ((outer * self.shape[axis] + cursor) * row_bytes) as usize;
                    result[target..target + row_bytes as usize]
                        .copy_from_slice(&source[start..start + row_bytes as usize]);
                    cursor += 1;
                }
            }
        }
        Ok(result)
    }
}
