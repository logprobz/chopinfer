//! Where each byte of a gathered, zero-padded tensor comes from.
//!
//! A gather takes ranges along one axis of a row-major source tensor and
//! concatenates them along that axis, then appends zeros along at most one
//! axis up to the stored shape. [`Gather::pieces`] maps any byte range of the
//! result, before widening, onto source byte ranges and zero runs, so a
//! record larger than a staging chunk is produced one part at a time.

use super::{SourceTensor, WeightTransformError, refuse};
use std::ops::Range;

/// One run of a gathered byte range: source bytes to copy, or zeros.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Piece {
    /// `length` bytes starting at `offset` within the source tensor.
    Source {
        offset: u64,
        length: u64,
    },
    Zero {
        length: u64,
    },
}

impl Piece {
    #[must_use]
    pub const fn length(self) -> u64 {
        match self {
            Self::Source { length, .. } | Self::Zero { length } => length,
        }
    }
}

/// The validated geometry of one gather.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gather {
    /// Source element bytes.
    element_bytes: u64,
    /// Whether each element widens from BF16 to F32 after gathering.
    widen: bool,
    /// Source bytes of one index along the selected axis.
    inner: u64,
    /// Source bytes of one whole outer index (every index of the axis).
    outer_stride: u64,
    /// Selected ranges along the axis, in order, and where each starts in
    /// the gathered axis.
    ranges: Vec<Range<u64>>,
    starts: Vec<u64>,
    /// Gathered indices along the selected axis.
    gathered_axis: u64,
    /// Whether the selection is the whole tensor in source order.
    whole: bool,
    /// Gathered and stored bytes of one block of the padded axis; equal
    /// when nothing is padded.
    live_block: u64,
    stored_block: u64,
    /// Stored elements and bytes before widening.
    stored_elements: u64,
}

fn checked(value: Option<u64>) -> Result<u64, WeightTransformError> {
    value.ok_or_else(|| WeightTransformError("gather size overflow".into()))
}

fn product(dims: &[u64]) -> Result<u64, WeightTransformError> {
    dims.iter()
        .try_fold(1_u64, |total, dim| total.checked_mul(*dim))
        .ok_or_else(|| WeightTransformError("gather size overflow".into()))
}

impl Gather {
    /// Checks a selection and stored shape against the source tensor.
    ///
    /// `axis: None` selects the whole tensor. Several ranges are allowed only
    /// when every dimension before the axis is 1, so their concatenation is
    /// also the order the ranges are read in.
    pub fn new(
        source: &SourceTensor,
        axis: Option<usize>,
        ranges: &[Range<u64>],
        stored: &[u64],
        widen: bool,
    ) -> Result<Self, WeightTransformError> {
        let element_bytes = source.element_bytes()?;
        let shape = if source.shape.is_empty() {
            vec![1]
        } else {
            source.shape.clone()
        };
        let (axis, ranges) = match axis {
            None if ranges.is_empty() => (0, std::iter::once(0..shape[0]).collect()),
            None => return refuse("ranges need an axis"),
            Some(axis) if axis < shape.len() && !ranges.is_empty() => (axis, ranges.to_vec()),
            Some(_) => return refuse("selection axis outside the tensor or no ranges"),
        };
        if ranges
            .iter()
            .any(|range| range.start >= range.end || range.end > shape[axis])
        {
            return refuse("selected range empty or outside its axis");
        }
        let outer = product(&shape[..axis])?;
        if ranges.len() > 1 && outer != 1 {
            return refuse("several ranges need every dimension before their axis to be 1");
        }
        let inner = checked(product(&shape[axis + 1..])?.checked_mul(element_bytes))?;
        let outer_stride = checked(inner.checked_mul(shape[axis]))?;
        let mut starts = Vec::with_capacity(ranges.len());
        let mut gathered_axis = 0_u64;
        for range in &ranges {
            starts.push(gathered_axis);
            gathered_axis = checked(gathered_axis.checked_add(range.end - range.start))?;
        }
        let mut gathered = shape.clone();
        gathered[axis] = gathered_axis;
        if stored.len() != gathered.len() {
            return refuse(format!(
                "stored shape {stored:?} and gathered shape {gathered:?} differ in rank"
            ));
        }
        let differs: Vec<_> = (0..stored.len())
            .filter(|&index| stored[index] != gathered[index])
            .collect();
        let (live_block, stored_block) = match differs[..] {
            [] => {
                let total = checked(product(&gathered)?.checked_mul(element_bytes))?;
                (total, total)
            }
            [padded] if stored[padded] > gathered[padded] => {
                let inner = checked(product(&stored[padded + 1..])?.checked_mul(element_bytes))?;
                (
                    checked(inner.checked_mul(gathered[padded]))?,
                    checked(inner.checked_mul(stored[padded]))?,
                )
            }
            _ => {
                return refuse(format!(
                    "stored shape {stored:?} must extend gathered shape {gathered:?} along at most one axis"
                ));
            }
        };
        let whole = ranges.len() == 1 && ranges[0] == (0..shape[axis]);
        Ok(Self {
            element_bytes,
            widen,
            inner,
            outer_stride,
            ranges,
            starts,
            gathered_axis,
            whole,
            live_block,
            stored_block,
            stored_elements: product(stored)?,
        })
    }

    #[must_use]
    pub const fn element_bytes(&self) -> u64 {
        self.element_bytes
    }

    #[must_use]
    pub const fn widens(&self) -> bool {
        self.widen
    }

    #[must_use]
    pub const fn stored_elements(&self) -> u64 {
        self.stored_elements
    }

    /// Stored bytes before widening.
    #[must_use]
    pub const fn stored_bytes(&self) -> u64 {
        self.stored_elements * self.element_bytes
    }

    /// Source bytes of gathered offset `offset`, and how many bytes stay
    /// contiguous in the source from there.
    fn gathered(&self, offset: u64) -> (u64, u64) {
        if self.whole {
            return (offset, u64::MAX);
        }
        let run = self.gathered_axis * self.inner;
        let (outer, within) = (offset / run, offset % run);
        let index = within / self.inner;
        let range = self.starts.partition_point(|start| *start <= index) - 1;
        let begin =
            self.ranges[range].start * self.inner + (within - self.starts[range] * self.inner);
        let end = self.ranges[range].end * self.inner;
        (outer * self.outer_stride + begin, end - begin)
    }

    /// The runs that make up stored bytes `range` before widening, in order.
    pub fn pieces(&self, range: Range<u64>) -> Result<Vec<Piece>, WeightTransformError> {
        if range.start > range.end || range.end > self.stored_bytes() {
            return refuse("gathered byte range outside the stored tensor");
        }
        let mut pieces: Vec<Piece> = Vec::new();
        let mut at = range.start;
        while at < range.end {
            let (block, within) = (at / self.stored_block, at % self.stored_block);
            let piece = if within >= self.live_block {
                Piece::Zero {
                    length: (self.stored_block - within).min(range.end - at),
                }
            } else {
                let (offset, contiguous) = self.gathered(block * self.live_block + within);
                Piece::Source {
                    offset,
                    length: contiguous.min(self.live_block - within).min(range.end - at),
                }
            };
            at += piece.length();
            match (pieces.last_mut(), piece) {
                (
                    Some(Piece::Source { offset, length }),
                    Piece::Source {
                        offset: next,
                        length: more,
                    },
                ) if *offset + *length == next => *length += more,
                (Some(Piece::Zero { length }), Piece::Zero { length: more }) => *length += more,
                _ => pieces.push(piece),
            }
        }
        Ok(pieces)
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    fn tensor(dtype: &str, shape: &[u64], element: u64) -> SourceTensor {
        SourceTensor {
            dtype: dtype.into(),
            shape: shape.to_vec(),
            byte_length: shape.iter().product::<u64>() * element,
        }
    }

    /// The gathered, padded bytes computed the direct way.
    fn reference(
        source: &[u8],
        shape: &[u64],
        element: u64,
        axis: usize,
        ranges: &[Range<u64>],
        stored: &[u64],
    ) -> Vec<u8> {
        let stride = shape[axis + 1..].iter().product::<u64>() * element;
        let outer = shape[..axis].iter().product::<u64>();
        let mut raw = Vec::new();
        for range in ranges {
            for row in 0..outer {
                let base = row * shape[axis] * stride;
                raw.extend_from_slice(
                    &source[(base + range.start * stride) as usize
                        ..(base + range.end * stride) as usize],
                );
            }
        }
        let mut gathered = shape.to_vec();
        gathered[axis] = ranges.iter().map(|r| r.end - r.start).sum();
        let Some(padded) = (0..stored.len()).find(|&a| stored[a] != gathered[a]) else {
            return raw;
        };
        let inner = stored[padded + 1..].iter().product::<u64>() * element;
        let (live, total) = (gathered[padded] * inner, stored[padded] * inner);
        let mut out = Vec::new();
        for block in raw.chunks_exact(live as usize) {
            out.extend_from_slice(block);
            out.resize(out.len() + (total - live) as usize, 0);
        }
        out
    }

    fn produce(gather: &Gather, source: &[u8], range: Range<u64>) -> Vec<u8> {
        let mut out = Vec::new();
        for piece in gather.pieces(range).unwrap() {
            match piece {
                Piece::Source { offset, length } => {
                    out.extend_from_slice(&source[offset as usize..(offset + length) as usize]);
                }
                Piece::Zero { length } => out.resize(out.len() + length as usize, 0),
            }
        }
        out
    }

    #[test]
    fn pieces_match_the_direct_gather_over_every_split() {
        type Case = (Vec<u64>, u64, usize, Vec<Range<u64>>, Vec<u64>);
        let cases: Vec<Case> = vec![
            (vec![8, 6], 1, 0, vec![2..5], vec![3, 6]),
            (vec![8, 6], 2, 0, vec![0..2, 4..5, 6..8], vec![5, 6]),
            (vec![8, 6], 2, 1, vec![1..4], vec![8, 3]),
            (vec![8, 6], 1, 0, vec![2..5], vec![7, 6]),
            (vec![8, 6], 2, 1, vec![2..4], vec![8, 5]),
            (vec![12, 1, 4], 2, 0, vec![0..2, 4..6, 8..12], vec![8, 1, 4]),
            (vec![3, 4, 5], 1, 1, vec![1..3], vec![3, 2, 7]),
            (vec![9], 2, 0, vec![0..9], vec![9]),
            (vec![6, 4], 1, 0, vec![0..6], vec![6, 4]),
        ];
        for (shape, element, axis, ranges, stored) in cases {
            let source_tensor = tensor("U8", &shape, element);
            let source: Vec<u8> = (0..source_tensor.byte_length)
                .map(|i| (i * 7 + 1) as u8)
                .collect();
            let expected = reference(&source, &shape, element, axis, &ranges, &stored);
            let gather = Gather::new(&source_tensor, Some(axis), &ranges, &stored, false).unwrap();
            assert_eq!(gather.stored_bytes(), expected.len() as u64);
            let total = expected.len() as u64;
            for start in 0..=total {
                for end in start..=total {
                    assert_eq!(
                        produce(&gather, &source, start..end),
                        &expected[start as usize..end as usize],
                        "{shape:?} axis {axis} {ranges:?} -> {stored:?} [{start}, {end})"
                    );
                }
            }
        }
    }

    #[test]
    fn whole_tensor_is_one_run() {
        let source = tensor("BF16", &[4, 8], 2);
        let gather = Gather::new(&source, None, &[], &[4, 8], false).unwrap();
        assert_eq!(
            gather.pieces(0..64).unwrap(),
            [Piece::Source {
                offset: 0,
                length: 64
            }]
        );
        assert_eq!(
            gather.pieces(3..5).unwrap(),
            [Piece::Source {
                offset: 3,
                length: 2
            }]
        );
    }

    #[test]
    fn refuses_inconsistent_selections() {
        let source = tensor("BF16", &[4, 8], 2);
        for (axis, ranges, stored) in [
            (Some(2), vec![0..1], vec![4, 8]),
            (Some(0), vec![0..5], vec![5, 8]),
            (Some(0), vec![2..2], vec![0, 8]),
            (Some(1), vec![0..2, 4..6], vec![4, 4]),
            (Some(0), vec![0..2], vec![1, 8]),
            (Some(0), vec![0..2], vec![3, 9]),
            (Some(0), vec![0..2], vec![2, 8, 1]),
            (None, vec![0..2], vec![2, 8]),
        ] {
            assert!(
                Gather::new(&source, axis, &ranges, &stored, false).is_err(),
                "{axis:?} {ranges:?} {stored:?}"
            );
        }
    }
}
