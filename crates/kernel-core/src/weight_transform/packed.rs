//! One section of a prepared precision operand, packed from its checkpoint
//! codes and scales by the shared prepared-weight rules: lossless row and K
//! slicing ([`crate::precision::packed::MatrixView::repack`]), bank placement,
//! then fragment order and scale tiles as the [`PackingPlan`] selects.
//!
//! The record's sources are the checkpoint tensors; each
//! [`PackedMatrixSource`] names the ones one bank matrix reads and the slice
//! it keeps. Every `(expert, projection)` of the plan is supplied once.

use super::{WeightRecord, WeightTransformError, refuse};
use crate::contract::{DType, WeightFormat};
use crate::precision::layout::PackingPlan;
use crate::precision::{DecodeOrder, NibbleOrder};
use std::ops::Range;

/// Which packed section the record stores.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum PackedSection {
    Codes,
    LocalScales,
    GlobalScales,
}

impl PackedSection {
    /// The section's position in [`PackedOperand::execute_sections`].
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Codes => 0,
            Self::LocalScales => 1,
            Self::GlobalScales => 2,
        }
    }
}

/// One source matrix of the bank and the slice of it the plan keeps.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct PackedMatrixSource {
    pub expert: u32,
    pub projection: u32,
    /// The checkpoint encoding and the logical `[rows, k]` of one matrix.
    pub format: WeightFormat,
    pub rows: u64,
    pub k: u64,
    /// Source coordinates of the first row and column, as the imported
    /// local-scale groups declare them.
    pub row_origin: u64,
    pub k_origin: u64,
    pub nibble_order: NibbleOrder,
    /// Record source indices of the codes, local scales and global scale.
    pub codes: u32,
    pub local: Option<u32>,
    pub local_dtype: Option<DType>,
    pub global: Option<u32>,
    /// When the source tensors hold a bank, the matrix index within it and
    /// the bank size; `(0, 1)` otherwise.
    pub source_expert: u64,
    pub source_experts: u64,
    /// Kept source rows, in order, and the kept K range.
    pub rows_kept: Vec<Range<u64>>,
    pub k_kept: Range<u64>,
}

/// The parameters of a `PACKED_OPERAND` record.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    feature = "precision-serde",
    derive(serde::Serialize, serde::Deserialize)
)]
#[cfg_attr(feature = "precision-serde", serde(deny_unknown_fields))]
pub struct PackedOperand {
    pub plan: PackingPlan,
    pub section: PackedSection,
    /// Checked before packing when scales apply before a BF16 MMA.
    pub decode_order: DecodeOrder,
    pub matrices: Vec<PackedMatrixSource>,
}

impl PackedOperand {
    pub fn encode(&self) -> Result<Vec<u8>, WeightTransformError> {
        #[cfg(feature = "precision-serde")]
        {
            serde_json::to_vec(self).map_err(|error| WeightTransformError(error.to_string()))
        }
        #[cfg(not(feature = "precision-serde"))]
        {
            refuse("packed operand parameters need kernel-core's precision-serde feature")
        }
    }

    pub fn decode(params: &[u8]) -> Result<Self, WeightTransformError> {
        #[cfg(feature = "precision-serde")]
        {
            serde_json::from_slice(params).map_err(|error| {
                WeightTransformError(format!("packed operand parameters: {error}"))
            })
        }
        #[cfg(not(feature = "precision-serde"))]
        {
            let _ = params;
            refuse("packed operand parameters need kernel-core's precision-serde feature")
        }
    }

    /// The stored section's bytes, after checking the bank is complete and
    /// every matrix names sources the record has.
    pub fn output_bytes(&self, record: &WeightRecord) -> Result<u64, WeightTransformError> {
        let error = |e: crate::precision::PrecisionError| {
            WeightTransformError(format!("packed operand {}: {}", e.field, e.reason))
        };
        self.plan.validate().map_err(error)?;
        if record.axis.is_some() || !record.ranges.is_empty() {
            return refuse("a packed operand keeps its slices in its matrices");
        }
        let mut seen = vec![false; (self.plan.experts * self.plan.projections) as usize];
        for matrix in &self.matrices {
            if matrix.expert >= self.plan.experts || matrix.projection >= self.plan.projections {
                return refuse("packed matrix outside the bank");
            }
            let index = (matrix.expert * self.plan.projections + matrix.projection) as usize;
            if std::mem::replace(&mut seen[index], true) {
                return refuse("packed matrix supplied twice");
            }
            for source in [Some(matrix.codes), matrix.local, matrix.global]
                .into_iter()
                .flatten()
            {
                if source as usize >= record.sources.len() {
                    return refuse("packed matrix names a missing source");
                }
            }
            if matrix.local.is_some() != matrix.local_dtype.is_some()
                || matrix.source_experts == 0
                || matrix.source_expert >= matrix.source_experts
                || matrix.rows_kept.is_empty()
                || matrix.k_kept.start >= matrix.k_kept.end
            {
                return refuse("packed matrix slice or scale description is inconsistent");
            }
        }
        if seen.contains(&false) {
            return refuse("not every bank matrix is supplied");
        }
        let sections = self.plan.section_bytes().map_err(error)?;
        Ok(match self.section {
            PackedSection::Codes => sections.codes,
            PackedSection::LocalScales => sections.local_scales,
            PackedSection::GlobalScales => sections.global_scales,
        })
    }
}

fn precision(error: crate::precision::PrecisionError) -> WeightTransformError {
    WeightTransformError(format!(
        "packed operand {} {}: {}",
        error.tensor, error.field, error.reason
    ))
}

impl PackedOperand {
    /// Packs the whole bank from the record's source tensors, in record
    /// order, and returns the stored section. Byte for byte this is the
    /// prepared-weight builder of `model-compiler-sdk`'s quantization packing.
    pub fn execute(
        &self,
        record: &WeightRecord,
        sources: &[&[u8]],
    ) -> Result<Vec<u8>, WeightTransformError> {
        let mut sections = self.execute_sections(record, sources)?;
        Ok(std::mem::take(&mut sections[self.section.index()]))
    }

    /// Every section of the packed bank, indexed by [`PackedSection::index`]:
    /// the records of one operand's sections differ only in the section
    /// they store, so one packing serves them all.
    pub fn execute_sections(
        &self,
        record: &WeightRecord,
        sources: &[&[u8]],
    ) -> Result<[Vec<u8>; 3], WeightTransformError> {
        use crate::contract::SwigluColumns;
        use crate::precision::packed::{MatrixSlice, MatrixView, ScaleBytes};
        use crate::transform::fragment::FragmentWeightLayout;
        use crate::transform::{ScaleTileLayout, Transform};

        let expected = self.output_bytes(record)?;
        if sources.len() != record.sources.len() {
            return refuse("packed operand source count differs from its record");
        }
        let plan = &self.plan;
        let (scale_rows, scale_columns) = plan.scale_shape();
        let mut codes = vec![0_u8; (plan.total_rows() * plan.code_bytes()) as usize];
        let mut scales = vec![
            if matches!(
                plan.format,
                WeightFormat::Fp8Block32x32 | WeightFormat::Mxfp4 | WeightFormat::Mxfp8
            ) {
                127
            } else {
                0
            };
            scale_rows * scale_columns * plan.scale_width()
        ];
        let mut globals = vec![
            0_u8;
            if plan.format == WeightFormat::Nvfp4 {
                if plan.per_expert_global() {
                    (plan.experts * plan.projections * 4) as usize
                } else {
                    4
                }
            } else {
                0
            }
        ];
        let mut scalar_local: Option<Vec<u8>> = None;
        let mut scalar_global: Option<Vec<u8>> = None;
        for matrix in &self.matrices {
            let part = |index: u32| -> Result<&[u8], WeightTransformError> {
                let bytes = sources[index as usize];
                let size = bytes.len() / matrix.source_experts as usize;
                if size * matrix.source_experts as usize != bytes.len() {
                    return refuse("bank source does not divide into its matrices");
                }
                let start = matrix.source_expert as usize * size;
                Ok(&bytes[start..start + size])
            };
            let global = match matrix.global {
                Some(index) => {
                    let bytes = sources[index as usize];
                    let at = if bytes.len() == 4 {
                        0
                    } else {
                        matrix.source_expert as usize * 4
                    };
                    Some(ScaleBytes {
                        dtype: DType::F32,
                        bytes: bytes.get(at..at + 4).ok_or_else(|| {
                            WeightTransformError("global scale out of bounds".into())
                        })?,
                    })
                }
                None => None,
            };
            let view = MatrixView {
                tensor: "packed operand",
                format: matrix.format,
                rows: matrix.rows as usize,
                k: matrix.k as usize,
                row_origin: matrix.row_origin as usize,
                k_origin: matrix.k_origin as usize,
                nibble_order: matrix.nibble_order,
                codes: part(matrix.codes)?,
                local: match (matrix.local, matrix.local_dtype) {
                    (Some(index), Some(dtype)) => Some(ScaleBytes {
                        dtype,
                        bytes: part(index)?,
                    }),
                    _ => None,
                },
                global,
            };
            view.validate().map_err(precision)?;
            if self.decode_order.scales_before_bf16_mma() {
                view.validate_bf16_operands(self.decode_order)
                    .map_err(precision)?;
            }
            let slice = MatrixSlice {
                rows: matrix
                    .rows_kept
                    .iter()
                    .flat_map(|range| range.start as usize..range.end as usize)
                    .collect(),
                k_start: matrix.k_kept.start as usize,
                k: (matrix.k_kept.end - matrix.k_kept.start) as usize,
                padded_rows: plan.rows as usize,
                padded_k: plan.k as usize,
                nibble_order: NibbleOrder::EvenLow,
            };
            // A 32x32 UE8M0 source is resident as its lossless MXFP8 rows:
            // expand the whole matrix, then slice, so a kept row reads its
            // own block's scales wherever the slice starts.
            let expanded = match (view.format, plan.format) {
                (WeightFormat::Fp8Block32x32, WeightFormat::Mxfp8) => Some(
                    crate::precision::format::registry::by_id(
                        crate::precision::format::FormatId::Fp8Block32,
                    )
                    .and_then(|plugin| plugin.repack_mxfp8_rows(view))
                    .map_err(precision)?,
                ),
                (source, resident) if source == resident => None,
                _ => return refuse("encoding differs from the prepared physical contract"),
            };
            let view = match &expanded {
                Some(rows) => rows.view("packed operand"),
                None => view,
            };
            let packed = view.repack(&slice).map_err(precision)?;
            let (rn, rk) = plan.groups();
            if (rn != u32::MAX && packed.row_origin % rn as usize != 0)
                || (rk != u32::MAX && packed.k_origin % rk as usize != 0)
            {
                return refuse("implementation requires an aligned source scale-group origin");
            }
            let (expert, projection) = (matrix.expert, matrix.projection);
            let index = (expert * plan.projections + projection) as usize;
            let row_bytes = plan.code_bytes() as usize;
            let base_row =
                (u64::from(expert) * u64::from(plan.projections) * u64::from(plan.rows)) as usize;
            let target_row = |r: usize| {
                base_row
                    + match plan.columns {
                        SwigluColumns::ConcatenatedHalves => {
                            projection as usize * plan.rows as usize + r
                        }
                        SwigluColumns::Interleaved => {
                            r * plan.projections as usize + projection as usize
                        }
                    }
            };
            let mut local = packed.scales;
            if let Some(dtype) = plan.local_scale_dtype {
                if packed.scale_dtype == DType::Bf16 && dtype == DType::F32 {
                    local = local
                        .chunks_exact(2)
                        .flat_map(|s| [0, 0, s[0], s[1]])
                        .collect();
                } else if packed.scale_dtype != dtype
                    && !(plan.format == WeightFormat::Nvfp4
                        && matches!(
                            (packed.scale_dtype, dtype),
                            (DType::Ue4M3, DType::Fp8E4M3) | (DType::Fp8E4M3, DType::Ue4M3)
                        ))
                {
                    return refuse("scale conversion would change stored values");
                }
            }
            if plan.format == WeightFormat::Fp8PerTensor {
                if scalar_local.as_ref().is_some_and(|old| old != &local) {
                    return refuse("fused per-tensor scales are not bitwise equal");
                }
                scalar_local = Some(local.clone());
            }
            if let Some(global) = &packed.global {
                if !plan.per_expert_global()
                    && scalar_global.as_ref().is_some_and(|old| old != global)
                {
                    return refuse(
                        "scalar-global implementation requires bitwise equal weight globals",
                    );
                }
                scalar_global = Some(global.clone());
                let start = if plan.per_expert_global() {
                    index * 4
                } else {
                    0
                };
                globals[start..start + 4].copy_from_slice(global);
            }
            for row in 0..plan.rows as usize {
                let target = target_row(row) * row_bytes;
                codes[target..target + row_bytes]
                    .copy_from_slice(&packed.codes[row * row_bytes..(row + 1) * row_bytes]);
            }
            if plan.format == WeightFormat::Fp8PerTensor {
                scales.copy_from_slice(&local);
            } else if plan.local_scale_dtype.is_some() {
                let width = plan.k.div_ceil(rk) as usize * plan.scale_width();
                let rows = plan.rows.div_ceil(rn) as usize;
                let scale_rows_per_expert = (plan.rows * plan.projections).div_ceil(rn) as usize;
                for row in 0..rows {
                    let target = if rn == 1 {
                        target_row(row)
                    } else {
                        expert as usize * scale_rows_per_expert + projection as usize * rows + row
                    };
                    scales[target * width..(target + 1) * width]
                        .copy_from_slice(&local[row * width..(row + 1) * width]);
                }
            }
        }
        let codes = match plan.fragment() {
            Some(format) => {
                let layout = FragmentWeightLayout {
                    rows: (plan.rows * plan.projections) as usize,
                    k: plan.k as usize,
                    format,
                };
                let bytes = layout.rows * layout.k * layout.element_bytes();
                let mut out = Vec::with_capacity(codes.len());
                for expert in codes.chunks_exact(bytes) {
                    out.extend(super::fragment::pack(layout, expert)?);
                }
                out
            }
            None => codes,
        };
        let scales = if plan.tiled_scales() {
            ScaleTileLayout {
                rows: scale_rows,
                k_blocks: scale_columns,
                tile_rows: 16,
                k_blocks_per_tile: 8,
            }
            .pack(&scales)
            .map_err(super::refusal)?
            .bytes
        } else {
            scales
        };
        let sections = [codes, scales, globals];
        if sections[self.section.index()].len() as u64 != expected {
            return refuse("packed section differs from the prepared contract");
        }
        Ok(sections)
    }
}

/// One run of consecutive kept source rows: `count` rows from source row
/// `source`, landing at section row `output`.
struct RowRun {
    source: u64,
    count: u64,
    output: u64,
}

impl PackedOperand {
    /// Bytes of one codes row, when the codes section can be produced by
    /// rows: one row-major matrix whose section rows are its kept source
    /// rows in order, then zero padding. Embedding tables and n-gram shards
    /// are such operands, and far larger than a staging chunk.
    #[must_use]
    pub fn code_rows(&self) -> Option<u64> {
        let plan = &self.plan;
        (self.section == PackedSection::Codes
            && plan.layout.id == "row-major"
            && plan.fragment().is_none()
            && plan.experts == 1
            && plan.projections == 1
            && self.matrices.len() == 1
            && self.matrices[0].format == plan.format
            && matches!(plan.groups().0, 1 | u32::MAX))
        .then(|| plan.code_bytes())
    }

    /// The kept source rows behind section rows `rows`, in runs.
    fn row_runs(&self, rows: Range<u64>) -> Vec<RowRun> {
        let mut runs = Vec::new();
        let mut output = 0_u64;
        for kept in &self.matrices[0].rows_kept {
            let length = kept.end - kept.start;
            let (start, end) = (rows.start.max(output), rows.end.min(output + length));
            if start < end {
                runs.push(RowRun {
                    source: kept.start + (start - output),
                    count: end - start,
                    output: start,
                });
            }
            output += length;
        }
        runs
    }

    fn code_bits(format: WeightFormat) -> u64 {
        match format {
            WeightFormat::F32 => 32,
            WeightFormat::Bf16 => 16,
            WeightFormat::Nvfp4 | WeightFormat::Mxfp4 => 4,
            _ => 8,
        }
    }

    /// Source ranges `(source, offset, length)` of section rows `rows`, for
    /// read-ahead: their code rows and, when scales go by row, scale rows.
    pub fn code_row_sources(
        &self,
        record: &WeightRecord,
        rows: Range<u64>,
    ) -> Vec<(usize, u64, u64)> {
        let matrix = &self.matrices[0];
        let code_row = matrix.k * Self::code_bits(matrix.format) / 8;
        let base = matrix.source_expert * matrix.rows * code_row;
        let mut ranges = Vec::new();
        for run in self.row_runs(rows) {
            ranges.push((
                matrix.codes as usize,
                base + run.source * code_row,
                run.count * code_row,
            ));
        }
        for index in [matrix.local, matrix.global].into_iter().flatten() {
            if let Some(source) = record.sources.get(index as usize) {
                ranges.push((index as usize, 0, source.byte_length));
            }
        }
        ranges
    }

    /// Codes section rows `rows`, reading only their source rows: the same
    /// bytes as [`Self::execute`]'s codes for those rows. Each part checks
    /// its own rows (codes and scales) as the whole-matrix check would.
    pub fn execute_code_rows(
        &self,
        record: &WeightRecord,
        rows: Range<u64>,
        read: &mut super::SourceRead<'_>,
    ) -> Result<Vec<u8>, WeightTransformError> {
        use crate::precision::packed::{MatrixSlice, MatrixView, ScaleBytes};
        let row_bytes = self
            .code_rows()
            .ok_or_else(|| WeightTransformError("packed operand is not row-streamable".into()))?;
        self.output_bytes(record)?;
        let plan = &self.plan;
        if rows.start > rows.end || rows.end > u64::from(plan.rows) {
            return refuse("packed rows outside the section");
        }
        let matrix = &self.matrices[0];
        if matrix.format != plan.format {
            return refuse("encoding differs from the prepared physical contract");
        }
        let bits = Self::code_bits(matrix.format);
        if matrix.rows == 0 || matrix.k == 0 || (bits == 4 && matrix.k % 2 != 0) {
            return refuse("empty matrix or odd packed row width");
        }
        let code_row = matrix
            .k
            .checked_mul(bits)
            .map(|v| v / 8)
            .ok_or_else(|| WeightTransformError("packed code row overflow".into()))?;
        let matrix_codes = matrix
            .rows
            .checked_mul(code_row)
            .ok_or_else(|| WeightTransformError("packed matrix size overflow".into()))?;
        if matrix_codes.checked_mul(matrix.source_experts)
            != Some(record.sources[matrix.codes as usize].byte_length)
        {
            return refuse("codes byte count disagrees with logical shape");
        }
        if let (Some(index), Some(dtype)) = (matrix.local, matrix.local_dtype) {
            let view = MatrixView {
                tensor: "packed operand",
                format: matrix.format,
                rows: matrix.rows as usize,
                k: matrix.k as usize,
                row_origin: matrix.row_origin as usize,
                k_origin: matrix.k_origin as usize,
                nibble_order: matrix.nibble_order,
                codes: &[],
                local: None,
                global: None,
            };
            let (scale_rows, scale_columns) = view.scale_shape().map_err(precision)?;
            let width = match dtype {
                DType::F32 => 4,
                DType::Bf16 => 2,
                _ => 1,
            };
            let expected = (scale_rows as u64)
                .checked_mul(scale_columns as u64)
                .and_then(|v| v.checked_mul(width))
                .and_then(|v| v.checked_mul(matrix.source_experts));
            if expected != Some(record.sources[index as usize].byte_length) {
                return refuse("local scale byte count disagrees with logical shape");
            }
        }
        if let Some(index) = matrix.global {
            let length = record.sources[index as usize].byte_length;
            if length != 4
                && matrix
                    .source_expert
                    .checked_add(1)
                    .and_then(|v| v.checked_mul(4))
                    .is_none_or(|end| end > length)
            {
                return refuse("global scale out of bounds");
            }
        }
        let whole = |index: u32, read: &mut super::SourceRead<'_>| {
            let length = record.sources[index as usize].byte_length;
            let mut bytes = vec![0; length as usize];
            read(index as usize, 0, &mut bytes).map(|()| bytes)
        };
        let global = matrix.global.map(|index| whole(index, read)).transpose()?;
        let (row_groups, k_groups) = plan.groups();
        let local_whole = match (matrix.local, row_groups) {
            (Some(index), u32::MAX) => Some(whole(index, read)?),
            _ => None,
        };
        let mut out = vec![0_u8; ((rows.end - rows.start) * row_bytes) as usize];
        for run in self.row_runs(rows.clone()) {
            let mut codes = vec![0; (run.count * code_row) as usize];
            read(
                matrix.codes as usize,
                matrix.source_expert * matrix_codes + run.source * code_row,
                &mut codes,
            )?;
            let local = match (matrix.local, &local_whole) {
                (Some(_), Some(bytes)) => {
                    let part = bytes.len() / matrix.source_experts as usize;
                    Some(bytes[matrix.source_expert as usize * part..][..part].to_vec())
                }
                (Some(index), None) => {
                    // One scale row per source row.
                    let total = record.sources[index as usize].byte_length
                        / matrix.source_experts
                        / matrix.rows;
                    let mut bytes = vec![0; (run.count * total) as usize];
                    read(
                        index as usize,
                        (matrix.source_expert * matrix.rows + run.source) * total,
                        &mut bytes,
                    )?;
                    Some(bytes)
                }
                (None, _) => None,
            };
            let global = global.as_ref().map(|bytes| {
                let at = if bytes.len() == 4 {
                    0
                } else {
                    matrix.source_expert as usize * 4
                };
                bytes[at..at + 4].to_vec()
            });
            let view = MatrixView {
                tensor: "packed operand",
                format: matrix.format,
                rows: run.count as usize,
                k: matrix.k as usize,
                row_origin: (matrix.row_origin + run.source) as usize,
                k_origin: matrix.k_origin as usize,
                nibble_order: matrix.nibble_order,
                codes: &codes,
                local: match (&local, matrix.local_dtype) {
                    (Some(bytes), Some(dtype)) => Some(ScaleBytes { dtype, bytes }),
                    _ => None,
                },
                global: global.as_deref().map(|bytes| ScaleBytes {
                    dtype: DType::F32,
                    bytes,
                }),
            };
            view.validate().map_err(precision)?;
            if self.decode_order.scales_before_bf16_mma() {
                view.validate_bf16_operands(self.decode_order)
                    .map_err(precision)?;
            }
            let packed = view
                .repack(&MatrixSlice {
                    rows: (0..run.count as usize).collect(),
                    k_start: matrix.k_kept.start as usize,
                    k: (matrix.k_kept.end - matrix.k_kept.start) as usize,
                    padded_rows: run.count as usize,
                    padded_k: plan.k as usize,
                    nibble_order: NibbleOrder::EvenLow,
                })
                .map_err(precision)?;
            if k_groups != u32::MAX && packed.k_origin % k_groups as usize != 0 {
                return refuse("implementation requires an aligned source scale-group origin");
            }
            let at = ((run.output - rows.start) * row_bytes) as usize;
            out[at..at + packed.codes.len()].copy_from_slice(&packed.codes);
        }
        Ok(out)
    }
}
