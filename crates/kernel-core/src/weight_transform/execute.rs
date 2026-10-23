//! Producing any byte range of a record from positional source reads.
//!
//! Row-major and fragment records are produced in parts, so a record larger
//! than a staging chunk never needs a whole-record buffer: plain copies read
//! straight into the output, fragments (MMA and MXFP4 MoE) read the rows of
//! the strips they cover and pack them into the output, and only strips cut
//! by the range edge go through a one-strip temporary. Scale tiles, packed
//! operands and inline bytes are small and are produced whole.
//!
//! Every part is checked against the record's payload domain
//! ([`super::payload`]) as it is gathered, before packing: a fragment's rows,
//! a copy's elements (a part that cuts an element reads the whole element).

use super::fragment::{pack_strips, strip_bytes};
use super::gather::{Gather, Piece};
use super::mxfp4::{Mxfp4Section, Mxfp4Strips};
use super::payload::{Payload, scale_bytes};
use super::{WeightRecord, WeightTransform, WeightTransformError, f32_to_bf16, refuse};
use crate::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use crate::transform::{ScaleTileLayout, Transform};
use std::ops::Range;

/// Reads `buffer.len()` bytes at `offset` within source tensor `source`.
pub type SourceRead<'a> = dyn FnMut(usize, u64, &mut [u8]) -> Result<(), WeightTransformError> + 'a;

/// How whole 16-row strips of a stored matrix pack.
#[derive(Clone, Copy, Debug)]
enum Strips {
    Mma(FragmentWeightLayout),
    Mxfp4(Mxfp4Strips),
}

impl Strips {
    fn rows(self) -> usize {
        match self {
            Self::Mma(layout) => layout.rows,
            Self::Mxfp4(strips) => strips.rows,
        }
    }

    fn row_bytes(self) -> usize {
        match self {
            Self::Mma(layout) => layout.k * layout.element_bytes(),
            Self::Mxfp4(strips) => strips.row_bytes,
        }
    }

    fn pack(self, rows: &[u8], first: usize, out: &mut [u8]) -> Result<(), WeightTransformError> {
        match self {
            Self::Mma(layout) => pack_strips(layout, rows, first, out),
            Self::Mxfp4(strips) => strips.pack_strips(rows, first, out),
        }
    }
}

#[derive(Clone, Debug)]
enum Kind {
    Rows(Gather),
    Narrow(Gather),
    Fragment {
        gather: Gather,
        layout: Strips,
        strip: u64,
    },
    /// A packed operand's codes, produced by rows of `row` bytes.
    PackedRows {
        row: u64,
    },
    Whole,
}

/// A record checked once and ready to produce any part of its bytes.
#[derive(Clone, Debug)]
pub struct PreparedRecord {
    record: WeightRecord,
    bytes: u64,
    kind: Kind,
    payload: Payload,
}

impl WeightRecord {
    /// Checks the record against its sources and states how it is produced.
    /// `declared` is the plan's length for it, which must match.
    pub fn prepare(self, declared: u64) -> Result<PreparedRecord, WeightTransformError> {
        let kind = match &self.transform {
            WeightTransform::Bundle(_) => Kind::Whole,
            WeightTransform::Copy | WeightTransform::CopyScales | WeightTransform::WidenBf16F32 => {
                Kind::Rows(self.gather()?)
            }
            WeightTransform::NarrowF32Bf16 => Kind::Narrow(self.gather()?),
            WeightTransform::Fragment(format) => {
                let layout = self.fragment_layout(*format)?;
                Kind::Fragment {
                    gather: self.gather()?,
                    strip: strip_bytes(layout)? as u64,
                    layout: Strips::Mma(layout),
                }
            }
            WeightTransform::Mxfp4Moe(section) => {
                let strips = self.mxfp4_strips(*section)?;
                Kind::Fragment {
                    gather: self.gather()?,
                    strip: strips.strip_bytes() as u64,
                    layout: Strips::Mxfp4(strips),
                }
            }
            WeightTransform::PackedOperand(operand) => match operand.code_rows() {
                Some(row) => Kind::PackedRows { row },
                None => Kind::Whole,
            },
            WeightTransform::ScaleTile { .. }
            | WeightTransform::Inline(_)
            | WeightTransform::SwapInnerAxes => Kind::Whole,
        };
        if !matches!(self.transform, WeightTransform::Bundle(_)) {
            let bytes = self.output_bytes()?;
            if bytes != declared {
                return refuse(format!(
                    "record produces {bytes} bytes, the plan places {declared}"
                ));
            }
            if let Kind::Fragment { gather, layout, .. } = &kind
                && gather.stored_bytes() != (layout.rows() * layout.row_bytes()) as u64
            {
                return refuse("fragment source bytes differ from its matrix");
            }
        }
        let payload = self.payload();
        Ok(PreparedRecord {
            record: self,
            bytes: declared,
            kind,
            payload,
        })
    }

    /// The domain of the bytes a gathering record stores: stated by its
    /// transform, or by an E4M3 source dtype.
    fn payload(&self) -> Payload {
        let dtype = match &self.sources[..] {
            [source] if self.transform.gathers() => source.dtype.as_str(),
            _ => return Payload::Unchecked,
        };
        match &self.transform {
            WeightTransform::CopyScales => {
                scale_bytes(dtype).map_or(Payload::Unchecked, |bytes| Payload::Scales { bytes })
            }
            WeightTransform::Mxfp4Moe(Mxfp4Section::Scales) => Payload::Mxfp4Scales,
            WeightTransform::Fragment(FragmentWeightFormat::Fp8E4M3) => Payload::E4m3,
            _ if dtype == "F8_E4M3" => Payload::E4m3,
            _ => Payload::Unchecked,
        }
    }
}

/// A covering read spans at most this many source bytes.
pub const COVERING_READ_BYTES: u64 = 1 << 20;
/// Only source pieces up to this long are fetched by covering reads; longer
/// ones are read on their own.
pub const COVERED_PIECE_BYTES: u64 = 64 << 10;

std::thread_local! {
    /// Each thread's covering-read buffer, kept between calls.
    static COVERING: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Fills `out` with gathered bytes `range` (before widening).
///
/// A selection along an inner axis (a K slice) is one short source piece
/// per row. Runs of short pieces in increasing source order are fetched
/// with one covering read of at most [`COVERING_READ_BYTES`] and copied out,
/// so a K-sliced matrix costs a few reads, not one per row. The bytes are
/// the same either way.
fn gathered(
    gather: &Gather,
    range: Range<u64>,
    out: &mut [u8],
    read: &mut SourceRead<'_>,
) -> Result<(), WeightTransformError> {
    let pieces = gather.pieces(range)?;
    let mut at = 0;
    let mut index = 0;
    while index < pieces.len() {
        let piece = pieces[index];
        let length = piece.length() as usize;
        let Piece::Source { offset, .. } = piece else {
            out[at..at + length].fill(0);
            at += length;
            index += 1;
            continue;
        };
        // The run of short pieces a covering read from `offset` can serve.
        let mut end = offset + length as u64;
        let mut last = index;
        let mut sources = 1;
        if piece.length() <= COVERED_PIECE_BYTES {
            for (next, candidate) in pieces.iter().enumerate().skip(index + 1) {
                match *candidate {
                    Piece::Zero { .. } => {}
                    Piece::Source {
                        offset: from,
                        length: count,
                    } if count <= COVERED_PIECE_BYTES
                        && from >= end
                        && from + count - offset <= COVERING_READ_BYTES =>
                    {
                        end = from + count;
                        sources += 1;
                    }
                    Piece::Source { .. } => break,
                }
                last = next;
            }
            // Trailing zero runs belong to the next step.
            while last > index && matches!(pieces[last], Piece::Zero { .. }) {
                last -= 1;
            }
        }
        if sources == 1 {
            read(0, offset, &mut out[at..at + length])?;
            at += length;
            index += 1;
            continue;
        }
        COVERING.with(|covering| -> Result<(), WeightTransformError> {
            let mut covering = covering.borrow_mut();
            covering.resize((end - offset) as usize, 0);
            read(0, offset, &mut covering)?;
            for piece in &pieces[index..=last] {
                let length = piece.length() as usize;
                match *piece {
                    Piece::Source { offset: from, .. } => {
                        let from = (from - offset) as usize;
                        out[at..at + length].copy_from_slice(&covering[from..from + length]);
                    }
                    Piece::Zero { .. } => out[at..at + length].fill(0),
                }
                at += length;
            }
            Ok(())
        })?;
        index = last + 1;
    }
    Ok(())
}

impl PreparedRecord {
    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub const fn record(&self) -> &WeightRecord {
        &self.record
    }

    /// Whether only [`Self::produce_whole`] can produce this record.
    #[must_use]
    pub const fn is_whole(&self) -> bool {
        matches!(self.kind, Kind::Whole)
    }

    /// The output granularity parts should be cut at to avoid repacking:
    /// one fragment strip, one widened element, or one byte.
    #[must_use]
    pub fn unit_bytes(&self) -> u64 {
        match &self.kind {
            Kind::Fragment { strip, .. } => *strip,
            Kind::Rows(gather) if gather.widens() => 4,
            Kind::Narrow(_) => 2,
            Kind::Rows(_) => self.payload.element_bytes(),
            Kind::PackedRows { row } => *row,
            Kind::Whole => self.bytes.max(1),
        }
    }

    /// The source ranges `(source, offset, length)` producing record bytes
    /// `range` reads, for read-ahead. Whole records read every source whole.
    pub fn source_ranges(
        &self,
        range: Range<u64>,
    ) -> Result<Vec<(usize, u64, u64)>, WeightTransformError> {
        let pieces = |gather: &Gather, range: Range<u64>| -> Result<_, WeightTransformError> {
            Ok(gather
                .pieces(range)?
                .into_iter()
                .filter_map(|piece| match piece {
                    Piece::Source { offset, length } => Some((0, offset, length)),
                    Piece::Zero { .. } => None,
                })
                .collect())
        };
        match &self.kind {
            Kind::Whole => Ok(self
                .record
                .sources
                .iter()
                .enumerate()
                .map(|(index, source)| (index, 0, source.byte_length))
                .collect()),
            Kind::Rows(gather) if gather.widens() => {
                pieces(gather, range.start / 4 * 2..range.end.div_ceil(4) * 2)
            }
            Kind::PackedRows { row } => {
                let WeightTransform::PackedOperand(operand) = &self.record.transform else {
                    unreachable!("packed rows come from packed operands");
                };
                Ok(operand
                    .code_row_sources(&self.record, range.start / row..range.end.div_ceil(*row)))
            }
            Kind::Narrow(gather) => pieces(gather, range.start / 2 * 4..range.end.div_ceil(2) * 4),
            Kind::Rows(gather) => pieces(gather, range),
            Kind::Fragment {
                gather,
                layout,
                strip,
            } => {
                let row_bytes = layout.row_bytes() as u64;
                let begin = range.start / strip * 16;
                let end = (range.end.div_ceil(*strip) * 16).min(layout.rows() as u64);
                pieces(gather, begin * row_bytes..end * row_bytes)
            }
        }
    }

    /// Writes record bytes `range` into `out`. `scratch` is reused between
    /// calls to hold source rows.
    pub fn produce(
        &self,
        range: Range<u64>,
        out: &mut [u8],
        read: &mut SourceRead<'_>,
        scratch: &mut Vec<u8>,
    ) -> Result<(), WeightTransformError> {
        if range.start > range.end
            || range.end > self.bytes
            || out.len() as u64 != range.end - range.start
        {
            return refuse("record range outside the record or the output");
        }
        match &self.kind {
            Kind::Whole => {
                let whole = self.produce_whole(read)?;
                out.copy_from_slice(&whole[range.start as usize..range.end as usize]);
                Ok(())
            }
            Kind::Rows(gather) if !gather.widens() => {
                let element = self.payload.element_bytes();
                let (first, last) = (
                    range.start / element * element,
                    range.end.div_ceil(element) * element,
                );
                if (first, last) == (range.start, range.end) {
                    gathered(gather, range, out, read)?;
                    return self.payload.check(out, first);
                }
                scratch.resize((last - first) as usize, 0);
                gathered(gather, first..last, scratch, read)?;
                self.payload.check(scratch, first)?;
                let skip = (range.start - first) as usize;
                out.copy_from_slice(&scratch[skip..skip + out.len()]);
                Ok(())
            }
            Kind::PackedRows { row } => {
                let WeightTransform::PackedOperand(operand) = &self.record.transform else {
                    unreachable!("packed rows come from packed operands");
                };
                let first = range.start / row;
                let rows = first..range.end.div_ceil(*row);
                if range.start.is_multiple_of(*row) && range.end.is_multiple_of(*row) {
                    return operand.execute_code_rows_into(&self.record, rows, out, read);
                }
                scratch.resize(((rows.end - rows.start) * row) as usize, 0);
                operand.execute_code_rows_into(&self.record, rows, scratch, read)?;
                let skip = (range.start - first * row) as usize;
                out.copy_from_slice(&scratch[skip..skip + out.len()]);
                Ok(())
            }
            Kind::Rows(gather) => {
                let (first, last) = (range.start / 4, range.end.div_ceil(4));
                scratch.resize(((last - first) * 2) as usize, 0);
                gathered(gather, first * 2..last * 2, scratch, read)?;
                let skip = (range.start - first * 4) as usize;
                let mut written = 0;
                for (index, value) in scratch.chunks_exact(2).enumerate() {
                    // A BF16 value is the upper half of the F32 of the same value.
                    let widened = [0, 0, value[0], value[1]];
                    let from = if index == 0 { skip } else { 0 };
                    let count = (4 - from).min(out.len() - written);
                    out[written..written + count].copy_from_slice(&widened[from..from + count]);
                    written += count;
                    if written == out.len() {
                        break;
                    }
                }
                Ok(())
            }
            Kind::Narrow(gather) => {
                let (first, last) = (range.start / 2, range.end.div_ceil(2));
                scratch.resize(((last - first) * 4) as usize, 0);
                gathered(gather, first * 4..last * 4, scratch, read)?;
                let skip = (range.start - first * 2) as usize;
                let mut written = 0;
                for (index, value) in scratch.chunks_exact(4).enumerate() {
                    let bits = u32::from_le_bytes(value.try_into().expect("four bytes"));
                    let narrowed = f32_to_bf16(bits).to_le_bytes();
                    let from = if index == 0 { skip } else { 0 };
                    let count = (2 - from).min(out.len() - written);
                    out[written..written + count].copy_from_slice(&narrowed[from..from + count]);
                    written += count;
                    if written == out.len() {
                        break;
                    }
                }
                Ok(())
            }
            Kind::Fragment {
                gather,
                layout,
                strip,
            } => {
                let strip = *strip;
                let row_bytes = layout.row_bytes() as u64;
                let mut rows_of = |first: u64,
                                   count: u64,
                                   scratch: &mut Vec<u8>|
                 -> Result<(), WeightTransformError> {
                    let begin = first * 16;
                    let end = ((first + count) * 16).min(layout.rows() as u64);
                    scratch.resize(((end - begin) * row_bytes) as usize, 0);
                    gathered(gather, begin * row_bytes..end * row_bytes, scratch, read)?;
                    self.payload.check(scratch, begin * row_bytes)
                };
                let mut at = range.start;
                let mut written = 0_usize;
                while at < range.end {
                    let index = at / strip;
                    let within = at % strip;
                    if within == 0 && range.end - at >= strip {
                        // Whole strips straight into the output.
                        let count = (range.end - at) / strip;
                        rows_of(index, count, scratch)?;
                        let length = (count * strip) as usize;
                        layout.pack(
                            scratch,
                            index as usize,
                            &mut out[written..written + length],
                        )?;
                        at += count * strip;
                        written += length;
                    } else {
                        // A strip cut by the range edge.
                        rows_of(index, 1, scratch)?;
                        let mut one = vec![0; strip as usize];
                        layout.pack(scratch, index as usize, &mut one)?;
                        let count = (strip - within).min(range.end - at) as usize;
                        out[written..written + count]
                            .copy_from_slice(&one[within as usize..within as usize + count]);
                        at += count as u64;
                        written += count;
                    }
                }
                Ok(())
            }
        }
    }

    /// Produces the whole record as sections: a packed operand's three
    /// sections (see [`super::PackedOperand::execute_sections`]), or the
    /// record's bytes as one section.
    pub fn produce_sections(
        &self,
        read: &mut SourceRead<'_>,
    ) -> Result<Vec<Vec<u8>>, WeightTransformError> {
        let record = &self.record;
        if let WeightTransform::PackedOperand(operand) = &record.transform {
            let mut sources = Vec::with_capacity(record.sources.len());
            for (index, source) in record.sources.iter().enumerate() {
                let mut bytes = vec![0; source.byte_length as usize];
                read(index, 0, &mut bytes)?;
                sources.push(bytes);
            }
            let views: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            return Ok(operand.execute_sections(record, &views)?.into());
        }
        Ok(vec![self.produce_whole(read)?])
    }

    /// The section of [`Self::produce_sections`] this record stores.
    #[must_use]
    pub fn section(&self) -> usize {
        match &self.record.transform {
            WeightTransform::PackedOperand(operand) => operand.section.index(),
            _ => 0,
        }
    }

    /// Produces the whole record, reading each source tensor whole.
    pub fn produce_whole(
        &self,
        read: &mut SourceRead<'_>,
    ) -> Result<Vec<u8>, WeightTransformError> {
        let record = &self.record;
        match &record.transform {
            WeightTransform::ScaleTile {
                tile_rows,
                k_blocks_per_tile,
            } => {
                let gather = record.gather()?;
                let mut grid = vec![0; gather.stored_bytes() as usize];
                gathered(&gather, 0..gather.stored_bytes(), &mut grid, read)?;
                self.payload.check(&grid, 0)?;
                let [.., rows, k_blocks] = record.stored_shape[..] else {
                    return refuse("scale tiles need a matrix");
                };
                Ok(ScaleTileLayout {
                    rows: rows as usize,
                    k_blocks: k_blocks as usize,
                    tile_rows: *tile_rows as usize,
                    k_blocks_per_tile: *k_blocks_per_tile as usize,
                }
                .pack(&grid)
                .map_err(super::refusal)?
                .bytes)
            }
            WeightTransform::PackedOperand(operand) => {
                let mut sources = Vec::with_capacity(record.sources.len());
                for (index, source) in record.sources.iter().enumerate() {
                    let mut bytes = vec![0; source.byte_length as usize];
                    read(index, 0, &mut bytes)?;
                    sources.push(bytes);
                }
                let views: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
                operand.execute(record, &views)
            }
            WeightTransform::Inline(bytes) => Ok(bytes.clone()),
            WeightTransform::SwapInnerAxes => {
                let source = &record.sources[0];
                let mut bytes = vec![0; source.byte_length as usize];
                read(0, 0, &mut bytes)?;
                Ok(swap_inner_axes(
                    &bytes,
                    &source.shape,
                    source.element_bytes()?,
                ))
            }
            WeightTransform::Bundle(id) => refuse(format!(
                "bundle transform {id} runs through the bundle's export"
            )),
            _ => {
                let mut out = vec![0; self.bytes as usize];
                self.produce(0..self.bytes, &mut out, read, &mut Vec::new())?;
                Ok(out)
            }
        }
    }
}

/// A row-major `[..., a, b]` tensor of `element`-byte elements as
/// `[..., b, a]`.
fn swap_inner_axes(bytes: &[u8], shape: &[u64], element: u64) -> Vec<u8> {
    let [.., a, b] = shape[..] else {
        unreachable!("checked when the record was prepared");
    };
    let (a, b, element) = (a as usize, b as usize, element as usize);
    let matrix = a * b * element;
    let mut out = vec![0; bytes.len()];
    for (from, to) in bytes.chunks_exact(matrix).zip(out.chunks_exact_mut(matrix)) {
        for i in 0..a {
            for j in 0..b {
                let source = (i * b + j) * element;
                let target = (j * a + i) * element;
                to[target..target + element].copy_from_slice(&from[source..source + element]);
            }
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::super::{COPY, SourceTensor};
    use super::*;
    use crate::transform::fragment::FragmentWeightFormat;

    fn produce_parts(record: &PreparedRecord, source: &[u8], cut: u64) -> Vec<u8> {
        let mut out = vec![0xee; record.byte_length() as usize];
        let mut scratch = Vec::new();
        let mut read = |_: usize, offset: u64, buffer: &mut [u8]| {
            buffer.copy_from_slice(&source[offset as usize..offset as usize + buffer.len()]);
            Ok(())
        };
        let mut at = 0;
        while at < record.byte_length() {
            let end = (at + cut).min(record.byte_length());
            record
                .produce(
                    at..end,
                    &mut out[at as usize..end as usize],
                    &mut read,
                    &mut scratch,
                )
                .unwrap();
            at = end;
        }
        out
    }

    /// Every part size gives the reference bytes: slice, pad, then pack.
    #[test]
    fn parts_equal_the_whole_reference() {
        let cases = [
            // An o_proj-like K slice, FP8 fragments.
            (
                FragmentWeightFormat::Fp8E4M3,
                "F8_E4M3",
                vec![40, 128],
                1,
                vec![32..96],
                vec![40, 64],
            ),
            // A shared expert padded along rows, BF16 fragments.
            (
                FragmentWeightFormat::Bf16,
                "BF16",
                vec![24, 32],
                0,
                vec![8..18],
                vec![32, 32],
            ),
            // A down projection padded along K.
            (
                FragmentWeightFormat::Fp8E4M3,
                "F8_E4M3",
                vec![20, 96],
                1,
                vec![0..48],
                vec![20, 64],
            ),
        ];
        for (format, dtype, shape, axis, ranges, stored) in cases {
            let element = if dtype == "BF16" { 2 } else { 1 };
            let tensor = SourceTensor {
                dtype: dtype.into(),
                byte_length: shape.iter().product::<u64>() * element,
                shape: shape.clone(),
            };
            // Valid E4M3 codes: a NaN code's low bit cleared.
            let source: Vec<u8> = (0..tensor.byte_length)
                .map(|i| (i * 37 + 11) as u8)
                .map(|b| if b & 0x7f == 0x7f { b ^ 1 } else { b })
                .collect();
            let record = WeightRecord {
                transform: WeightTransform::Fragment(format),
                sources: vec![tensor.clone()],
                axis: Some(axis),
                ranges: ranges.clone(),
                stored_shape: stored.clone(),
            };
            let bytes = record.output_bytes().unwrap();
            let prepared = record.prepare(bytes).unwrap();
            let gather =
                Gather::new(&tensor, Some(axis as usize), &ranges, &stored, false).unwrap();
            let mut raw = vec![0; gather.stored_bytes() as usize];
            let mut read = |_: usize, offset: u64, buffer: &mut [u8]| {
                buffer.copy_from_slice(&source[offset as usize..offset as usize + buffer.len()]);
                Ok(())
            };
            gathered(&gather, 0..gather.stored_bytes(), &mut raw, &mut read).unwrap();
            let expected = FragmentWeightLayout {
                rows: stored[0] as usize,
                k: stored[1] as usize,
                format,
            }
            .pack(&raw)
            .unwrap()
            .bytes;
            for cut in [1, 7, 100, 511, 512, 1000, 4096, u64::MAX / 2] {
                assert_eq!(
                    produce_parts(&prepared, &source, cut),
                    expected,
                    "{shape:?} {cut}"
                );
            }
        }
    }

    #[test]
    fn widening_parts_equal_the_widened_whole() {
        let tensor = SourceTensor {
            dtype: "BF16".into(),
            shape: vec![6, 5],
            byte_length: 60,
        };
        let source: Vec<u8> = (0..60).map(|i| (i * 13 + 1) as u8).collect();
        let record = WeightRecord {
            transform: WeightTransform::WidenBf16F32,
            sources: vec![tensor],
            axis: Some(0),
            ranges: vec![1..4],
            stored_shape: vec![4, 5],
        };
        let prepared = record.prepare(80).unwrap();
        let mut expected: Vec<u8> = source[10..40]
            .chunks_exact(2)
            .flat_map(|v| [0, 0, v[0], v[1]])
            .collect();
        expected.resize(80, 0);
        for cut in 1..=80 {
            assert_eq!(produce_parts(&prepared, &source, cut), expected, "{cut}");
        }
    }

    /// A row-major packed operand (the shape of an embedding table or an
    /// n-gram shard): its codes section in parts, each reading only the
    /// source rows it covers, equal to the section packed whole.
    #[allow(clippy::too_many_arguments)]
    fn row_major_operand(
        format: crate::contract::WeightFormat,
        dtype: &str,
        rows: u64,
        k: u64,
        element: u64,
        rows_kept: Vec<Range<u64>>,
        k_kept: Range<u64>,
        padded_rows: u32,
        local: Option<(crate::contract::DType, u64)>,
    ) -> (WeightRecord, Vec<Vec<u8>>) {
        use super::super::packed::{PackedMatrixSource, PackedOperand, PackedSection};
        use crate::contract::SwigluColumns;
        use crate::precision::layout::PackingPlan;
        use crate::precision::{DecodeOrder, NibbleOrder, WeightLayout};
        let code_bytes = if format == crate::contract::WeightFormat::Nvfp4 {
            rows * k / 2
        } else {
            rows * k * element
        };
        let codes: Vec<u8> = (0..code_bytes)
            .map(|i| ((i * 37 + 5) % 251) as u8 & 0x7e)
            .collect();
        let mut sources = vec![codes];
        let mut tensors = vec![SourceTensor {
            dtype: dtype.into(),
            shape: vec![rows, k],
            byte_length: code_bytes,
        }];
        let global = format == crate::contract::WeightFormat::Nvfp4;
        if let Some((local_dtype, bytes)) = local {
            sources.push(if local_dtype == crate::contract::DType::Bf16 {
                vec![0x80, 0x3f]
            } else {
                (0..bytes).map(|i| 0x30 + (i % 9) as u8).collect()
            });
            tensors.push(SourceTensor {
                dtype: "U8".into(),
                shape: vec![bytes],
                byte_length: bytes,
            });
        }
        if global {
            sources.push(1.5_f32.to_le_bytes().to_vec());
            tensors.push(SourceTensor {
                dtype: "F32".into(),
                shape: vec![],
                byte_length: 4,
            });
        }
        let operand = PackedOperand {
            local_encoding: None,
            plan: PackingPlan {
                layout: WeightLayout::row_major(),
                format,
                experts: 1,
                projections: 1,
                rows: padded_rows,
                k: (k_kept.end - k_kept.start) as u32,
                columns: SwigluColumns::ConcatenatedHalves,
                local_scale_dtype: local.map(|(dtype, _)| dtype),
            },
            section: PackedSection::Codes,
            decode_order: DecodeOrder::ScaleThenCode,
            matrices: vec![PackedMatrixSource {
                expert: 0,
                projection: 0,
                format,
                rows,
                k,
                row_origin: 0,
                k_origin: 0,
                nibble_order: NibbleOrder::EvenLow,
                codes: 0,
                local: local.map(|_| 1),
                local_dtype: local.map(|(dtype, _)| dtype),
                global: global.then_some(2),
                source_expert: 0,
                source_experts: 1,
                rows_kept,
                k_kept,
            }],
        };
        let record = WeightRecord {
            transform: WeightTransform::PackedOperand(Box::new(operand)),
            sources: tensors,
            axis: None,
            ranges: vec![],
            stored_shape: vec![],
        };
        (record, sources)
    }

    #[test]
    fn row_major_packed_parts_reject_mismatched_source_sizes() {
        use crate::contract::{DType, WeightFormat};
        for (source, extra) in [(0, 32), (1, 1), (1, 4)] {
            let (mut record, mut sources) = row_major_operand(
                WeightFormat::Nvfp4,
                "U8",
                4,
                64,
                1,
                vec![0..2],
                0..64,
                2,
                Some((DType::Fp8E4M3, 16)),
            );
            sources[source].extend(vec![0x38; extra]);
            record.sources[source].byte_length += extra as u64;
            let WeightTransform::PackedOperand(operand) = &record.transform else {
                unreachable!()
            };
            let views: Vec<_> = sources.iter().map(Vec::as_slice).collect();
            assert!(operand.execute_sections(&record, &views).is_err());
            let mut reads = 0;
            let result = operand.execute_code_rows(&record, 0..1, &mut |index, offset, out| {
                reads += 1;
                out.copy_from_slice(&sources[index][offset as usize..offset as usize + out.len()]);
                Ok(())
            });
            assert!(
                result.is_err(),
                "source {source} with {extra} extra bytes was accepted"
            );
            assert_eq!(reads, 0, "reject before reading any source");
        }
    }

    #[test]
    fn row_major_packed_parts_read_only_their_rows() {
        use crate::contract::{DType, WeightFormat};
        let cases = [
            // A BF16 lookup table's rank rows, padded.
            row_major_operand(
                WeightFormat::Bf16,
                "BF16",
                600,
                32,
                2,
                vec![100..400],
                0..32,
                320,
                None,
            ),
            // An FP8 n-gram shard with its per-tensor BF16 scale.
            row_major_operand(
                WeightFormat::Fp8PerTensor,
                "F8_E4M3",
                700,
                48,
                1,
                vec![0..650],
                0..48,
                650,
                Some((DType::Bf16, 2)),
            ),
            // NVFP4 rows with per-row E4M3 scales, a K slice and padding.
            row_major_operand(
                WeightFormat::Nvfp4,
                "U8",
                300,
                64,
                1,
                vec![20..90, 150..260],
                0..64,
                200,
                Some((DType::Fp8E4M3, 300 * 4)),
            ),
        ];
        for (record, sources) in cases {
            let bytes = record.output_bytes().unwrap();
            let prepared = record.clone().prepare(bytes).unwrap();
            let views: Vec<&[u8]> = sources.iter().map(Vec::as_slice).collect();
            let WeightTransform::PackedOperand(operand) = &record.transform else {
                unreachable!()
            };
            let whole = operand.execute(&record, &views).unwrap();
            let row = bytes / u64::from(operand.plan.rows);
            for cut in [row, 7 * row + 3, 64 * row, bytes / 3 + 1, bytes] {
                let read_bytes = std::cell::Cell::new(0_u64);
                let mut read = |index: usize, offset: u64, buffer: &mut [u8]| {
                    read_bytes.set(read_bytes.get() + buffer.len() as u64);
                    buffer.copy_from_slice(
                        &sources[index][offset as usize..offset as usize + buffer.len()],
                    );
                    Ok(())
                };
                let mut out = vec![0xee; bytes as usize];
                let mut at = 0;
                let mut largest = 0;
                while at < bytes {
                    let end = (at + cut).min(bytes);
                    let before = read_bytes.get();
                    prepared
                        .produce(
                            at..end,
                            &mut out[at as usize..end as usize],
                            &mut read,
                            &mut Vec::new(),
                        )
                        .unwrap();
                    largest = largest.max(read_bytes.get() - before);
                    at = end;
                }
                assert_eq!(out, whole, "cut {cut}");
                // Each part reads at most its rows (plus a row at each edge),
                // their scale rows, and the operand's scalar scales.
                let source_rows = record.sources[0].shape[0];
                let scale_row = match sources.get(1) {
                    Some(scales) if scales.len() as u64 >= source_rows => {
                        scales.len() as u64 / source_rows
                    }
                    _ => 0,
                };
                let source_row = sources[0].len() as u64 / source_rows + scale_row;
                let rows = cut.div_ceil(row) + 1;
                assert!(
                    largest <= rows * source_row + 64,
                    "cut {cut}: read {largest}"
                );
            }
        }
    }

    /// MXFP4 expert slices, rows (gate and up) and K (down), produced in
    /// parts, equal the reference bank packer over the gathered slice.
    #[test]
    fn mxfp4_parts_equal_the_reference_packer() {
        use crate::transform::Mxfp4MoeFragmentLayout;
        use crate::weight_transform::Mxfp4Section;
        for (section, shape, axis, ranges, stored) in [
            (
                Mxfp4Section::Codes,
                vec![64, 64],
                0_u32,
                vec![16..48],
                vec![32, 64],
            ),
            (
                Mxfp4Section::Codes,
                vec![32, 128],
                1,
                vec![32..96],
                vec![32, 64],
            ),
            (
                Mxfp4Section::Scales,
                vec![64, 8],
                0,
                vec![32..64],
                vec![32, 8],
            ),
            (
                Mxfp4Section::Scales,
                vec![48, 8],
                1,
                vec![4..8],
                vec![48, 4],
            ),
        ] {
            let tensor = SourceTensor {
                dtype: "U8".into(),
                byte_length: shape.iter().product::<u64>(),
                shape: shape.clone(),
            };
            // Scales stay in their domain, at most 252.
            let source: Vec<u8> = (0..tensor.byte_length)
                .map(|i| (i * 29 + 3) as u8)
                .map(|b| {
                    if section == Mxfp4Section::Scales {
                        b % 253
                    } else {
                        b
                    }
                })
                .collect();
            let gather =
                Gather::new(&tensor, Some(axis as usize), &ranges, &stored, false).unwrap();
            let mut raw = vec![0; gather.stored_bytes() as usize];
            let mut read = |_: usize, offset: u64, buffer: &mut [u8]| {
                buffer.copy_from_slice(&source[offset as usize..offset as usize + buffer.len()]);
                Ok(())
            };
            gathered(&gather, 0..gather.stored_bytes(), &mut raw, &mut read).unwrap();
            let (rows, row_bytes) = (stored[0] as usize, stored[1] as usize);
            let expected = match section {
                Mxfp4Section::Codes => Mxfp4MoeFragmentLayout {
                    experts: 1,
                    rows_per_expert: rows,
                    k: row_bytes * 2,
                }
                .pack_codes(&raw),
                Mxfp4Section::Scales => Mxfp4MoeFragmentLayout {
                    experts: 1,
                    rows_per_expert: rows,
                    k: row_bytes * 32,
                }
                .pack_scales(&raw),
            }
            .unwrap()
            .bytes;
            let record = WeightRecord {
                transform: WeightTransform::Mxfp4Moe(section),
                sources: vec![tensor],
                axis: Some(axis),
                ranges,
                stored_shape: stored,
            };
            let prepared = record.prepare(expected.len() as u64).unwrap();
            for cut in [1, 5, 64, 100, 512, 1000, u64::MAX / 2] {
                assert_eq!(produce_parts(&prepared, &source, cut), expected, "{cut}");
            }
        }
        let odd = WeightRecord {
            transform: WeightTransform::Mxfp4Moe(Mxfp4Section::Codes),
            sources: vec![SourceTensor {
                dtype: "U8".into(),
                shape: vec![24, 32],
                byte_length: 768,
            }],
            axis: None,
            ranges: vec![],
            stored_shape: vec![24, 32],
        };
        assert!(odd.output_bytes().is_err());
    }

    #[test]
    fn narrowing_parts_equal_the_rounded_whole() {
        let values: [u32; 12] = [
            0x3f80_8000, // tie, even: down
            0x3f81_8000, // tie, odd: up
            0x3f80_8001,
            0x7f7f_ffff, // rounds to infinity
            0xff80_0000,
            0x7fc0_0001,
            0x7f80_0001, // signalling NaN becomes quiet
            0x0000_0001,
            0x8000_7fff,
            0x4049_0fdb,
            0,
            0xc2f6_e979,
        ];
        let source: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let record = WeightRecord {
            transform: WeightTransform::NarrowF32Bf16,
            sources: vec![SourceTensor {
                dtype: "F32".into(),
                shape: vec![3, 4],
                byte_length: 48,
            }],
            axis: Some(0),
            ranges: vec![1..3],
            stored_shape: vec![3, 4],
        };
        let prepared = record.clone().prepare(24).unwrap();
        let mut expected: Vec<u8> = values[4..]
            .iter()
            .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
            .collect();
        expected.resize(24, 0);
        assert_eq!(&expected[..6], &[0x80, 0xff, 0xc0, 0x7f, 0xc0, 0x7f]);
        assert_eq!(f32_to_bf16(0x3f80_8000), 0x3f80);
        assert_eq!(f32_to_bf16(0x3f81_8000), 0x3f82);
        assert_eq!(f32_to_bf16(0x7f7f_ffff), 0x7f80);
        for cut in 1..=24 {
            assert_eq!(produce_parts(&prepared, &source, cut), expected, "{cut}");
        }
        let widened = WeightRecord {
            sources: vec![SourceTensor {
                dtype: "BF16".into(),
                shape: vec![3, 4],
                byte_length: 24,
            }],
            ..record
        };
        assert!(widened.output_bytes().is_err());
    }

    /// Swapping the last two axes of `[2, 3, 5]` and `[4, 3]` tensors of 1-,
    /// 2- and 4-byte elements matches the index arithmetic of a
    /// convolution weight's im2col order, whole and in parts; a stored shape
    /// other than the swapped one, a selection or a vector is refused.
    #[test]
    fn swapped_inner_axes_follow_the_transposed_index() {
        for (shape, dtype, element) in [
            (vec![2_u64, 3, 5], "BF16", 2_usize),
            (vec![4, 3], "U8", 1),
            (vec![2, 3, 5], "F32", 4),
        ] {
            let elements = shape.iter().product::<u64>() as usize;
            let source: Vec<u8> = (0..elements * element).map(|i| (i * 7 + 1) as u8).collect();
            let [.., a, b] = shape[..] else {
                unreachable!()
            };
            let (a, b) = (a as usize, b as usize);
            let mut expected = vec![0; source.len()];
            for o in 0..elements / (a * b) {
                for i in 0..a {
                    for j in 0..b {
                        let from = ((o * a + i) * b + j) * element;
                        let to = ((o * b + j) * a + i) * element;
                        expected[to..to + element].copy_from_slice(&source[from..from + element]);
                    }
                }
            }
            let mut stored = shape.clone();
            let last = stored.len() - 1;
            stored.swap(last - 1, last);
            let record = WeightRecord {
                transform: WeightTransform::SwapInnerAxes,
                sources: vec![SourceTensor {
                    dtype: dtype.into(),
                    shape: shape.clone(),
                    byte_length: source.len() as u64,
                }],
                axis: None,
                ranges: vec![],
                stored_shape: stored,
            };
            let prepared = record.clone().prepare(source.len() as u64).unwrap();
            assert!(prepared.is_whole());
            let mut read = |_: usize, offset: u64, out: &mut [u8]| {
                out.copy_from_slice(&source[offset as usize..offset as usize + out.len()]);
                Ok(())
            };
            assert_eq!(prepared.produce_whole(&mut read).unwrap(), expected);
            assert_eq!(produce_parts(&prepared, &source, 7), expected);
            let wrong = WeightRecord {
                stored_shape: shape.clone(),
                ..record.clone()
            };
            assert!(wrong.prepare(source.len() as u64).is_err());
            let selected = WeightRecord {
                axis: Some(0),
                ranges: vec![0..1],
                ..record
            };
            assert!(selected.prepare(source.len() as u64).is_err());
        }
        // A zero or overflowing dimension, or bytes that are not the shape's,
        // are refused before any production.
        for (shape, byte_length) in [
            (vec![0_u64, 2], 4_u64),
            (vec![2, 0], 4),
            (vec![3, 0, 2], 8),
            (vec![u64::MAX, 2], 4),
            (vec![2, 3], 8),
        ] {
            let mut stored = shape.clone();
            let last = stored.len() - 1;
            stored.swap(last - 1, last);
            let record = WeightRecord {
                transform: WeightTransform::SwapInnerAxes,
                sources: vec![SourceTensor {
                    dtype: "BF16".into(),
                    shape: shape.clone(),
                    byte_length,
                }],
                axis: None,
                ranges: vec![],
                stored_shape: stored,
            };
            assert!(
                record.prepare(byte_length).is_err(),
                "{shape:?} {byte_length}"
            );
        }
        let vector = WeightRecord {
            transform: WeightTransform::SwapInnerAxes,
            sources: vec![SourceTensor {
                dtype: "U8".into(),
                shape: vec![4],
                byte_length: 4,
            }],
            axis: None,
            ranges: vec![],
            stored_shape: vec![4],
        };
        assert!(vector.prepare(4).is_err());
        assert_eq!(
            WeightTransform::decode(super::super::SWAP_INNER_AXES, &[]).unwrap(),
            WeightTransform::SwapInnerAxes
        );
        assert!(WeightTransform::decode(super::super::SWAP_INNER_AXES, &[0; 4]).is_err());
    }

    #[test]
    fn inline_records_store_their_parameters() {
        let bytes: Vec<u8> = (0..100).collect();
        let transform = WeightTransform::decode(
            crate::weight_transform::INLINE,
            &WeightTransform::Inline(bytes.clone())
                .encode_params()
                .unwrap(),
        )
        .unwrap();
        let record = WeightRecord {
            transform,
            sources: vec![],
            axis: None,
            ranges: vec![],
            stored_shape: vec![],
        };
        let prepared = record.clone().prepare(100).unwrap();
        assert!(prepared.is_whole());
        assert_eq!(produce_parts(&prepared, &[], 7), bytes);
        assert!(record.prepare(99).is_err());
        assert!(WeightTransform::decode(crate::weight_transform::INLINE, &[]).is_err());
        assert!(
            WeightTransform::decode(
                crate::weight_transform::INLINE,
                &vec![0; crate::weight_transform::INLINE_BYTES_MAX + 1]
            )
            .is_err()
        );
    }

    /// A K slice is one short piece per row: covering reads fetch many rows
    /// at once, and the bytes equal the row-by-row gather at every part size.
    #[test]
    fn k_slices_are_read_with_covering_reads() {
        for (rows, row_bytes, keep, stored) in [
            (4096_u64, 1024_u64, 256..512, vec![4096, 256]),
            (64, 64 << 10, 1024..3072, vec![64, 2048]),
            (40, 48, 8..40, vec![48, 32]),
        ] {
            let tensor = SourceTensor {
                dtype: "U8".into(),
                shape: vec![rows, row_bytes],
                byte_length: rows * row_bytes,
            };
            let source: Vec<u8> = (0..tensor.byte_length)
                .map(|i| (i * 31 + (i >> 10)) as u8)
                .collect();
            let mut expected = Vec::new();
            for row in 0..rows {
                let base = (row * row_bytes) as usize;
                expected.extend_from_slice(
                    &source[base + keep.start as usize..base + keep.end as usize],
                );
            }
            expected.resize(stored.iter().product::<u64>() as usize, 0);
            let record = WeightRecord {
                transform: WeightTransform::Copy,
                sources: vec![tensor],
                axis: Some(1),
                ranges: vec![keep.clone()],
                stored_shape: stored,
            };
            let prepared = record.prepare(expected.len() as u64).unwrap();
            for cut in [7, 4096, 1 << 20, u64::MAX / 2] {
                let reads = std::cell::Cell::new(0_u64);
                let mut read = |_: usize, offset: u64, buffer: &mut [u8]| {
                    reads.set(reads.get() + 1);
                    buffer
                        .copy_from_slice(&source[offset as usize..offset as usize + buffer.len()]);
                    Ok(())
                };
                let mut out = vec![0xee; expected.len()];
                let mut at = 0;
                while at < out.len() as u64 {
                    let end = (at + cut).min(out.len() as u64);
                    prepared
                        .produce(
                            at..end,
                            &mut out[at as usize..end as usize],
                            &mut read,
                            &mut Vec::new(),
                        )
                        .unwrap();
                    at = end;
                }
                assert_eq!(out, expected, "{rows}x{row_bytes} cut {cut}");
                if cut == u64::MAX / 2 {
                    // One covering read per COVERING_READ_BYTES of source.
                    let covering = (rows * row_bytes).div_ceil(COVERING_READ_BYTES) + 1;
                    assert!(
                        reads.get() <= covering.max(1) || row_bytes > COVERED_PIECE_BYTES,
                        "{rows}x{row_bytes}: {} reads",
                        reads.get()
                    );
                }
            }
        }
    }

    /// Covering reads at dtype, order and size boundaries: multi-byte
    /// elements, ranges out of source order, pieces at the covered-piece
    /// limit and runs that reach or straddle the covering-read cap. The bytes
    /// equal the direct gather at every part size, no read passes the cap,
    /// and a whole-record read takes as few reads as the cap allows.
    #[test]
    fn covering_reads_hold_at_dtype_order_and_size_boundaries() {
        let cap = COVERING_READ_BYTES;
        let piece = COVERED_PIECE_BYTES;
        // (dtype, element bytes, shape, axis, ranges, stored, reads of the
        // whole record)
        type Case = (
            &'static str,
            u64,
            [u64; 2],
            u32,
            Vec<Range<u64>>,
            [u64; 2],
            u64,
        );
        let cases: Vec<Case> = vec![
            // 350-byte BF16 pieces 1400 bytes apart: 749 rows per read.
            ("BF16", 2, [3000, 700], 1, vec![175..350], [3000, 175], 5),
            // F32 pieces with padding between rows: 512 rows per read.
            ("F32", 4, [1024, 512], 1, vec![64..192], [1024, 160], 2),
            // Ranges out of source order: a covering read stops where the
            // next piece lies behind it, so [40..48], [0..8, 16..24, 56..64]
            // and [8..16] are read apart.
            (
                "BF16",
                2,
                [64, 48],
                0,
                vec![40..48, 0..8, 16..24, 56..64, 8..16],
                [40, 48],
                3,
            ),
            // Four covered pieces span exactly the cap, the fifth passes it.
            ("U8", 1, [16, 5 * piece], 1, vec![0..piece], [16, piece], 4),
            // A piece one byte over the limit is read alone.
            (
                "U8",
                1,
                [16, 5 * piece],
                1,
                vec![0..piece + 1],
                [16, piece + 1],
                16,
            ),
            // One byte longer rows: the fourth piece straddles the cap.
            (
                "U8",
                1,
                [16, 5 * piece + 1],
                1,
                vec![0..piece],
                [16, piece],
                6,
            ),
        ];
        for (dtype, element, shape, axis, ranges, stored, whole_reads) in cases {
            let [rows, columns] = shape;
            let row_bytes = columns * element;
            let tensor = SourceTensor {
                dtype: dtype.into(),
                shape: shape.to_vec(),
                byte_length: rows * row_bytes,
            };
            let source: Vec<u8> = (0..tensor.byte_length)
                .map(|i| (i * 31 + (i >> 10)) as u8)
                .collect();
            let mut expected = Vec::new();
            if axis == 0 {
                for range in &ranges {
                    expected.extend_from_slice(
                        &source
                            [(range.start * row_bytes) as usize..(range.end * row_bytes) as usize],
                    );
                }
            } else {
                for row in 0..rows {
                    let base = row * row_bytes;
                    for range in &ranges {
                        expected.extend_from_slice(
                            &source[(base + range.start * element) as usize
                                ..(base + range.end * element) as usize],
                        );
                    }
                    expected.resize(((row + 1) * stored[1] * element) as usize, 0);
                }
            }
            expected.resize((stored[0] * stored[1] * element) as usize, 0);
            let label = format!("{dtype} {shape:?} axis {axis} {ranges:?}");
            let record = WeightRecord {
                transform: WeightTransform::Copy,
                sources: vec![tensor],
                axis: Some(axis),
                ranges,
                stored_shape: stored.to_vec(),
            };
            let prepared = record.prepare(expected.len() as u64).unwrap();
            for cut in [7, 4096, 1 << 20, u64::MAX / 2] {
                let reads = std::cell::RefCell::new(Vec::new());
                let mut read = |_: usize, offset: u64, buffer: &mut [u8]| {
                    reads.borrow_mut().push(buffer.len() as u64);
                    buffer
                        .copy_from_slice(&source[offset as usize..offset as usize + buffer.len()]);
                    Ok(())
                };
                let mut out = vec![0xee; expected.len()];
                let mut at = 0;
                while at < out.len() as u64 {
                    let end = (at + cut).min(out.len() as u64);
                    prepared
                        .produce(
                            at..end,
                            &mut out[at as usize..end as usize],
                            &mut read,
                            &mut Vec::new(),
                        )
                        .unwrap();
                    at = end;
                }
                assert!(out == expected, "{label} cut {cut}: bytes differ");
                let reads = reads.into_inner();
                assert!(
                    reads.iter().all(|&length| length <= cap),
                    "{label} cut {cut}: a read passes the cap"
                );
                if cut == u64::MAX / 2 {
                    assert_eq!(reads.len() as u64, whole_reads, "{label}: {reads:?}");
                }
            }
        }
    }

    #[test]
    fn a_wrong_declared_length_is_refused() {
        let record = WeightRecord {
            transform: WeightTransform::Copy,
            sources: vec![SourceTensor {
                dtype: "BF16".into(),
                shape: vec![4],
                byte_length: 8,
            }],
            axis: None,
            ranges: vec![],
            stored_shape: vec![4],
        };
        assert!(record.clone().prepare(9).is_err());
        assert!(record.prepare(8).is_ok());
    }

    fn produce_all(record: WeightRecord, source: &[u8]) -> Result<Vec<u8>, WeightTransformError> {
        let bytes = record.output_bytes()?;
        record
            .prepare(bytes)?
            .produce_whole(&mut |_, offset, buffer| {
                buffer.copy_from_slice(&source[offset as usize..offset as usize + buffer.len()]);
                Ok(())
            })
    }

    fn matrix(transform: WeightTransform, dtype: &str, rows: u64, k: u64) -> WeightRecord {
        WeightRecord {
            transform,
            sources: vec![SourceTensor {
                dtype: dtype.into(),
                shape: vec![rows, k],
                byte_length: rows * k,
            }],
            axis: None,
            ranges: vec![],
            stored_shape: vec![rows, k],
        }
    }

    /// E4M3 codes and scales (by transform or source dtype) and MXFP4 strip
    /// scales outside their domain refuse the record, naming the stored
    /// byte; the same bytes in an unchecked domain are stored unchanged.
    #[test]
    fn stored_codes_and_scales_outside_their_domain_are_refused() {
        let fp8 = WeightTransform::Fragment(FragmentWeightFormat::Fp8E4M3);
        let mxfp4_scales = WeightTransform::Mxfp4Moe(Mxfp4Section::Scales);
        let tile = WeightTransform::ScaleTile {
            tile_rows: 16,
            k_blocks_per_tile: 8,
        };
        for (record, bad, refused) in [
            (matrix(fp8.clone(), "F8_E4M3", 16, 32), 0x7f, true),
            (matrix(fp8, "U8", 16, 32), 0xff, true),
            (matrix(WeightTransform::Copy, "F8_E4M3", 4, 8), 0xff, true),
            (matrix(tile, "F8_E4M3", 16, 8), 0x7f, true),
            (matrix(mxfp4_scales.clone(), "U8", 16, 2), 253, true),
            (matrix(mxfp4_scales, "U8", 16, 2), 255, true),
            (matrix(WeightTransform::Copy, "U8", 4, 8), 0xff, false),
            (
                matrix(WeightTransform::Mxfp4Moe(Mxfp4Section::Codes), "U8", 16, 32),
                0xff,
                false,
            ),
        ] {
            let length = record.sources[0].byte_length as usize;
            let mut source = vec![0x38; length];
            produce_all(record.clone(), &source).unwrap();
            source[length - 3] = bad;
            let result = produce_all(record.clone(), &source);
            if refused {
                let error = result.unwrap_err().0;
                assert!(
                    error.contains(&format!("stored byte {}", length - 3)),
                    "{:?}: {error}",
                    record.transform
                );
            } else {
                result.unwrap();
            }
        }
    }

    /// Checked scales store what a copy stores, in parts of every size, and
    /// a negative or non-finite scale refuses every part holding it.
    #[test]
    fn checked_scales_refuse_negative_or_nonfinite_values_in_every_part() {
        assert_eq!(WeightTransform::CopyScales.id(), COPY);
        let params = WeightTransform::CopyScales.encode_params().unwrap();
        assert_eq!(
            WeightTransform::decode(COPY, &params).unwrap(),
            WeightTransform::CopyScales
        );
        assert_eq!(
            WeightTransform::decode(COPY, &[]).unwrap(),
            WeightTransform::Copy
        );
        assert!(WeightTransform::decode(COPY, &2_u32.to_le_bytes()).is_err());
        let scales =
            |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let record = |transform| WeightRecord {
            transform,
            sources: vec![SourceTensor {
                dtype: "F32".into(),
                shape: vec![3, 4],
                byte_length: 48,
            }],
            axis: Some(1),
            ranges: vec![1..4],
            stored_shape: vec![3, 4],
        };
        let checked = record(WeightTransform::CopyScales).prepare(48).unwrap();
        let copied = record(WeightTransform::Copy).prepare(48).unwrap();
        assert_eq!(checked.unit_bytes(), 4);
        let mut values = [0.5_f32; 12];
        values[0] = -1.0;
        let valid = scales(&values);
        for cut in 1..=13 {
            assert_eq!(
                produce_parts(&checked, &valid, cut),
                produce_parts(&copied, &valid, cut)
            );
        }
        for bad in [-2.0, f32::NAN, f32::INFINITY, -0.0] {
            values[5] = bad;
            let source = scales(&values);
            let mut read = |_: usize, offset: u64, buffer: &mut [u8]| {
                buffer.copy_from_slice(&source[offset as usize..offset as usize + buffer.len()]);
                Ok(())
            };
            for at in 16..20 {
                let error = checked
                    .produce(at..at + 1, &mut [0], &mut read, &mut Vec::new())
                    .unwrap_err()
                    .0;
                assert!(error.contains("stored byte 16"), "{error}");
            }
            checked
                .produce(20..48, &mut [0; 28], &mut read, &mut Vec::new())
                .unwrap();
        }
        let mut bytes = matrix(WeightTransform::CopyScales, "U8", 4, 8);
        assert!(bytes.clone().prepare(32).is_err());
        bytes.sources[0].dtype = "F8_E4M3".into();
        bytes.prepare(32).unwrap();
    }

    /// Two groups of 48 rows from one stacked 96x64 E4M3 source with 32x32
    /// UE8M0 blocks: the block of rows 32..64 straddles the groups. Each
    /// group packs as MXFP8 rows, every row carrying its own block's scales.
    #[test]
    fn block32_groups_pack_as_their_lossless_mxfp8_rows() {
        use super::super::packed::{PackedMatrixSource, PackedOperand, PackedSection};
        use crate::contract::{DType, SwigluColumns, WeightFormat};
        use crate::precision::layout::PackingPlan;
        use crate::precision::{DecodeOrder, NibbleOrder, WeightLayout};
        let (rows, k, group) = (96_u64, 64_u64, 48_u64);
        let codes: Vec<u8> = (0..rows * k)
            .map(|i| ((i * 37 + 5) % 251) as u8 & 0x7e)
            .collect();
        let blocks: Vec<u8> = (0..(rows / 32) * (k / 32)).map(|i| 120 + i as u8).collect();
        let plan = |layout: &str| PackingPlan {
            layout: WeightLayout {
                id: layout.into(),
                version: 1,
            },
            format: WeightFormat::Mxfp8,
            experts: 2,
            projections: 1,
            rows: group as u32,
            k: k as u32,
            columns: SwigluColumns::ConcatenatedHalves,
            local_scale_dtype: Some(DType::Ue8M0),
        };
        assert!(plan("row-major").validate().is_err());
        let operand = |section| PackedOperand {
            local_encoding: None,
            plan: plan("row-major-groups"),
            section,
            decode_order: DecodeOrder::ExactScaledOperand,
            matrices: (0..2)
                .map(|g| PackedMatrixSource {
                    expert: g as u32,
                    projection: 0,
                    format: WeightFormat::Fp8Block32x32,
                    rows,
                    k,
                    row_origin: 0,
                    k_origin: 0,
                    nibble_order: NibbleOrder::EvenLow,
                    codes: 0,
                    local: Some(1),
                    local_dtype: Some(DType::Ue8M0),
                    global: None,
                    source_expert: 0,
                    source_experts: 1,
                    rows_kept: vec![g * group..(g + 1) * group],
                    k_kept: 0..k,
                })
                .collect(),
        };
        let record = |section| WeightRecord {
            transform: WeightTransform::PackedOperand(Box::new(operand(section))),
            sources: vec![
                SourceTensor {
                    dtype: "F8_E4M3".into(),
                    shape: vec![rows, k],
                    byte_length: rows * k,
                },
                SourceTensor {
                    dtype: "U8".into(),
                    shape: vec![rows / 32, k / 32],
                    byte_length: blocks.len() as u64,
                },
            ],
            axis: None,
            ranges: vec![],
            stored_shape: vec![],
        };
        let sources = [codes.as_slice(), blocks.as_slice()];
        let packed = operand(PackedSection::Codes)
            .execute_sections(&record(PackedSection::Codes), &sources)
            .unwrap();
        assert_eq!(packed[0], codes);
        let expected: Vec<u8> = (0..rows)
            .flat_map(|row| {
                let block = (row / 32) as usize * 2;
                blocks[block..block + 2].to_vec()
            })
            .collect();
        assert_eq!(packed[1], expected);
        // The expansion is never row-streamed from the 32x32 source.
        assert_eq!(operand(PackedSection::Codes).code_rows(), None);
    }
}
