//! The closed set of weight transforms a direct checkpoint load executes.
//!
//! A bundle's weight plan (`runtime-bundle-abi`'s `weight_plan`) places each
//! record's output at an arena offset and names the checkpoint tensors it
//! reads, a selection of those tensors, the stored shape and one transform
//! from this set. The host executes records without model code. Each
//! transform is exact: its bytes equal what the compiled rank-image path
//! writes for the same piece.
//!
//! Gather transforms read one source tensor. The record's selection takes
//! ranges along one axis of it and concatenates them along that axis; the
//! result is zero-padded along at most one axis to the stored shape.
//! [`WeightTransform::Copy`] stores those elements as they are,
//! [`WeightTransform::CopyScales`] stores quantization scales as they are
//! after checking each one,
//! [`WeightTransform::WidenBf16F32`] widens each BF16 element to the F32 of
//! the same value first, [`WeightTransform::Fragment`] packs the stored
//! `[rows, k]` matrix into MMA fragment order and
//! [`WeightTransform::ScaleTile`] packs a `[rows, k_blocks]` byte grid into
//! block-scaled MMA scale tiles.
//!
//! [`WeightTransform::Mxfp4Moe`] packs a `[rows, row_bytes]` byte matrix of
//! MXFP4 codes or scales into MoE fragment order, and
//! [`WeightTransform::NarrowF32Bf16`] rounds each F32 element to the nearest
//! BF16, ties to even.
//!
//! [`WeightTransform::SwapInnerAxes`] stores one whole tensor with its last
//! two axes swapped, `[..., a, b]` as `[..., b, a]`: a convolution weight's
//! `(out, channel, tap)` in the `(out, tap, channel)` order im2col columns
//! take.
//!
//! [`WeightTransform::Inline`] reads no tensor: its parameters are its bytes,
//! for a small piece a bundle takes from a checkpoint document and checks by
//! digest.
//!
//! [`WeightTransform::PackedOperand`] packs one prepared precision operand
//! (codes, then its local and global scale tensors when it has them) through
//! the shared prepared-weight rules, and stores one section of the result.
//!
//! Concatenation (gate and up halves, q, k and v rows) is two records at
//! adjacent arena offsets, not a transform.
//!
//! Identifiers from [`BUNDLE_TRANSFORM_FIRST`] up belong to the bundle: the
//! host hands such a record's whole source tensors to the bundle's own
//! transform callback. A bundle uses one only for a layout this set cannot
//! express.
//!
//! Records check the payloads whose domain they know while they stream (see
//! [`payload`]): E4M3 codes and scales, MXFP4 strip scales and the scales a
//! [`WeightTransform::CopyScales`] record states. A packed operand checks its
//! codes and scales through the shared precision rules. A payload outside its
//! domain refuses the record before any device reads it.
//!
//! Today's packers in [`crate::transform`] stay the reference: the tests of
//! this module compare every output with them.

pub mod execute;
pub mod fragment;
pub mod gather;
pub mod mxfp4;
pub mod packed;
pub mod payload;

use crate::transform::ScaleTileLayout;
use crate::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use std::fmt;
use std::ops::Range;

pub use execute::{PreparedRecord, SourceRead};
pub use mxfp4::{Mxfp4Section, Mxfp4Strips};
pub use packed::{PackedMatrixSource, PackedOperand, PackedSection, ScaleEncoding};

/// Gather, zero pad, store unchanged. Parameters: none, or one
/// little-endian `u32` payload, [`COPY_SCALES`].
pub const COPY: u32 = 1;
/// Gather, widen BF16 to F32, zero pad.
pub const WIDEN_BF16_F32: u32 = 2;
/// Gather, zero pad, pack into MMA fragment order. Parameter: one
/// little-endian `u32` format, [`FRAGMENT_BF16_16X16`] or
/// [`FRAGMENT_FP8_E4M3_16X32`].
pub const FRAGMENT: u32 = 3;
/// Gather, zero pad, pack 1-byte scales into MMA scale tiles. Parameters:
/// little-endian `u32` tile rows, then `u32` K blocks per tile.
pub const SCALE_TILE: u32 = 4;
/// One prepared precision operand section. Parameter: the JSON of a
/// [`PackedOperand`].
pub const PACKED_OPERAND: u32 = 5;
/// Gather one expert matrix's MXFP4 bytes, pack into MoE fragment order.
/// Parameter: one little-endian `u32`, [`MXFP4_CODES`] or [`MXFP4_SCALES`].
pub const MXFP4_MOE: u32 = 6;
/// Gather, round F32 to BF16 to nearest with ties to even, zero pad.
pub const NARROW_F32_BF16: u32 = 7;
/// Store the parameters, at most [`INLINE_BYTES_MAX`] bytes; no sources.
pub const INLINE: u32 = 8;
/// Store one whole tensor with its last two axes swapped. No parameters.
pub const SWAP_INNER_AXES: u32 = 9;
/// The `COPY` payload of quantization scales: F32, BF16 or E4M3 elements,
/// each finite with a clear sign bit.
pub const COPY_SCALES: u32 = 1;
/// The `MXFP4_MOE` section parameters.
pub const MXFP4_CODES: u32 = 1;
pub const MXFP4_SCALES: u32 = 2;
/// The longest inline record.
pub const INLINE_BYTES_MAX: usize = 64 << 10;
/// The first identifier a bundle may define for its own transform callback.
pub const BUNDLE_TRANSFORM_FIRST: u32 = 0x8000_0000;

/// The `FRAGMENT` format parameter of `bf16_mma_fragments_16x16_v1`.
pub const FRAGMENT_BF16_16X16: u32 = 1;
/// The `FRAGMENT` format parameter of `fp8_e4m3_mma_fragments_16x32_v1`.
pub const FRAGMENT_FP8_E4M3_16X32: u32 = 2;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightTransformError(pub String);

impl fmt::Display for WeightTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for WeightTransformError {}

pub(crate) fn refuse<T>(reason: impl Into<String>) -> Result<T, WeightTransformError> {
    Err(WeightTransformError(reason.into()))
}

/// One transform of the closed set, with its parameters.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WeightTransform {
    Copy,
    /// A copy of quantization scales, each checked as it streams.
    CopyScales,
    WidenBf16F32,
    Fragment(FragmentWeightFormat),
    ScaleTile {
        tile_rows: u32,
        k_blocks_per_tile: u32,
    },
    PackedOperand(Box<PackedOperand>),
    Mxfp4Moe(Mxfp4Section),
    NarrowF32Bf16,
    Inline(Vec<u8>),
    SwapInnerAxes,
    /// A bundle-defined transform, executed by the bundle's callback.
    Bundle(u32),
}

impl WeightTransform {
    #[must_use]
    pub fn id(&self) -> u32 {
        match self {
            Self::Copy | Self::CopyScales => COPY,
            Self::WidenBf16F32 => WIDEN_BF16_F32,
            Self::Fragment(_) => FRAGMENT,
            Self::ScaleTile { .. } => SCALE_TILE,
            Self::PackedOperand(_) => PACKED_OPERAND,
            Self::Mxfp4Moe(_) => MXFP4_MOE,
            Self::NarrowF32Bf16 => NARROW_F32_BF16,
            Self::Inline(_) => INLINE,
            Self::SwapInnerAxes => SWAP_INNER_AXES,
            Self::Bundle(id) => *id,
        }
    }

    /// The parameter bytes a plan record carries. A bundle transform's
    /// parameters are its own and are passed through unread.
    pub fn encode_params(&self) -> Result<Vec<u8>, WeightTransformError> {
        Ok(match self {
            Self::Copy
            | Self::WidenBf16F32
            | Self::NarrowF32Bf16
            | Self::SwapInnerAxes
            | Self::Bundle(_) => Vec::new(),
            Self::CopyScales => COPY_SCALES.to_le_bytes().to_vec(),
            Self::Mxfp4Moe(section) => match section {
                Mxfp4Section::Codes => MXFP4_CODES,
                Mxfp4Section::Scales => MXFP4_SCALES,
            }
            .to_le_bytes()
            .to_vec(),
            Self::Inline(bytes) => bytes.clone(),
            Self::Fragment(format) => match format {
                FragmentWeightFormat::Bf16 => FRAGMENT_BF16_16X16,
                FragmentWeightFormat::Fp8E4M3 => FRAGMENT_FP8_E4M3_16X32,
            }
            .to_le_bytes()
            .to_vec(),
            Self::ScaleTile {
                tile_rows,
                k_blocks_per_tile,
            } => [tile_rows.to_le_bytes(), k_blocks_per_tile.to_le_bytes()].concat(),
            Self::PackedOperand(operand) => operand.encode()?,
        })
    }

    /// Reads a record's transform. Identifiers outside the set and below
    /// [`BUNDLE_TRANSFORM_FIRST`] are refused.
    pub fn decode(id: u32, params: &[u8]) -> Result<Self, WeightTransformError> {
        let words = |count: usize| -> Result<Vec<u32>, WeightTransformError> {
            if params.len() != count * 4 {
                return refuse(format!(
                    "transform {id} takes {count} u32 parameters, not {} bytes",
                    params.len()
                ));
            }
            Ok(params
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("four bytes")))
                .collect())
        };
        Ok(match id {
            COPY if params.is_empty() => Self::Copy,
            COPY => match words(1)?[0] {
                COPY_SCALES => Self::CopyScales,
                other => return refuse(format!("unknown copy payload {other}")),
            },
            WIDEN_BF16_F32 => {
                words(0)?;
                Self::WidenBf16F32
            }
            FRAGMENT => match words(1)?[0] {
                FRAGMENT_BF16_16X16 => Self::Fragment(FragmentWeightFormat::Bf16),
                FRAGMENT_FP8_E4M3_16X32 => Self::Fragment(FragmentWeightFormat::Fp8E4M3),
                other => return refuse(format!("unknown fragment format {other}")),
            },
            SCALE_TILE => {
                let words = words(2)?;
                Self::ScaleTile {
                    tile_rows: words[0],
                    k_blocks_per_tile: words[1],
                }
            }
            PACKED_OPERAND => Self::PackedOperand(Box::new(PackedOperand::decode(params)?)),
            MXFP4_MOE => match words(1)?[0] {
                MXFP4_CODES => Self::Mxfp4Moe(Mxfp4Section::Codes),
                MXFP4_SCALES => Self::Mxfp4Moe(Mxfp4Section::Scales),
                other => return refuse(format!("unknown MXFP4 section {other}")),
            },
            NARROW_F32_BF16 => {
                words(0)?;
                Self::NarrowF32Bf16
            }
            INLINE if !params.is_empty() && params.len() <= INLINE_BYTES_MAX => {
                Self::Inline(params.to_vec())
            }
            INLINE => return refuse("an inline record holds 1 byte to 64 KiB"),
            SWAP_INNER_AXES => {
                words(0)?;
                Self::SwapInnerAxes
            }
            id if id >= BUNDLE_TRANSFORM_FIRST => Self::Bundle(id),
            other => return refuse(format!("unknown weight transform {other}")),
        })
    }

    /// Whether the record reads exactly one tensor through a [`gather::Gather`].
    #[must_use]
    pub fn gathers(&self) -> bool {
        matches!(
            self,
            Self::Copy
                | Self::CopyScales
                | Self::WidenBf16F32
                | Self::Fragment(_)
                | Self::ScaleTile { .. }
                | Self::Mxfp4Moe(_)
                | Self::NarrowF32Bf16
        )
    }
}

/// One source tensor as its checkpoint header declares it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceTensor {
    /// The safetensors dtype name, for example `BF16` or `F8_E4M3`.
    pub dtype: String,
    pub shape: Vec<u64>,
    pub byte_length: u64,
}

impl SourceTensor {
    /// Bytes of one element; zero-dimensional tensors hold one element.
    pub fn element_bytes(&self) -> Result<u64, WeightTransformError> {
        let elements = self.shape.iter().product::<u64>().max(1);
        if self.byte_length == 0 || !self.byte_length.is_multiple_of(elements) {
            return refuse("tensor bytes are not a whole number of elements");
        }
        Ok(self.byte_length / elements)
    }
}

/// A record as the host executes it: the transform, what it reads, and the
/// stored shape. Offsets are the plan's business, not this type's.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightRecord {
    pub transform: WeightTransform,
    pub sources: Vec<SourceTensor>,
    /// The axis the ranges select along, or `None` for the whole tensor.
    pub axis: Option<u32>,
    pub ranges: Vec<Range<u64>>,
    pub stored_shape: Vec<u64>,
}

impl WeightRecord {
    /// The bytes this record produces, after checking that its transform,
    /// sources, selection and stored shape agree.
    pub fn output_bytes(&self) -> Result<u64, WeightTransformError> {
        match &self.transform {
            WeightTransform::PackedOperand(operand) => operand.output_bytes(self),
            WeightTransform::Bundle(_) => refuse("a bundle transform declares its own length"),
            WeightTransform::Inline(bytes) => {
                if !self.sources.is_empty() || self.axis.is_some() || !self.ranges.is_empty() {
                    return refuse("an inline record reads nothing");
                }
                Ok(bytes.len() as u64)
            }
            WeightTransform::SwapInnerAxes => {
                let [source] = &self.sources[..] else {
                    return refuse("swapping axes reads exactly one tensor");
                };
                if self.axis.is_some() || !self.ranges.is_empty() {
                    return refuse("swapping axes reads the whole tensor");
                }
                let [lead @ .., a, b] = &source.shape[..] else {
                    return refuse("swapping axes needs at least two dimensions");
                };
                let swapped = [lead, &[*b, *a][..]].concat();
                if self.stored_shape != swapped {
                    return refuse(
                        "the stored shape is not the source's with its last two axes swapped",
                    );
                }
                // Every dimension holds elements, and the tensor's bytes are a
                // whole number of them that fits this host's addresses.
                let elements = source
                    .shape
                    .iter()
                    .try_fold(1_u64, |total, dim| {
                        (*dim != 0).then(|| total.checked_mul(*dim)).flatten()
                    })
                    .ok_or_else(|| {
                        WeightTransformError(
                            "swapping axes needs nonzero, bounded dimensions".into(),
                        )
                    })?;
                let element = source.element_bytes()?;
                if elements.checked_mul(element) != Some(source.byte_length)
                    || usize::try_from(source.byte_length).is_err()
                {
                    return refuse("swapped tensor bytes differ from its shape");
                }
                Ok(source.byte_length)
            }
            transform => {
                let gather = self.gather()?;
                match transform {
                    WeightTransform::Copy | WeightTransform::CopyScales => {
                        Ok(gather.stored_bytes())
                    }
                    WeightTransform::WidenBf16F32 => Ok(gather.stored_elements() * 4),
                    WeightTransform::NarrowF32Bf16 => Ok(gather.stored_elements() * 2),
                    WeightTransform::Mxfp4Moe(section) => {
                        Ok(self.mxfp4_strips(*section)?.packed_bytes()? as u64)
                    }
                    WeightTransform::Fragment(format) => Ok(self
                        .fragment_layout(*format)?
                        .packed_bytes()
                        .map_err(refusal)?
                        as u64),
                    WeightTransform::ScaleTile {
                        tile_rows,
                        k_blocks_per_tile,
                    } => {
                        let layout = self.scale_tile_layout(*tile_rows, *k_blocks_per_tile)?;
                        Ok((layout.rows.div_ceil(layout.tile_rows)
                            * layout.k_blocks.div_ceil(layout.k_blocks_per_tile)
                            * 128) as u64)
                    }
                    _ => unreachable!("gathering transforms only"),
                }
            }
        }
    }

    /// The gather of a single-source transform.
    pub fn gather(&self) -> Result<gather::Gather, WeightTransformError> {
        let [source] = &self.sources[..] else {
            return refuse("a gathering transform reads exactly one tensor");
        };
        let widen = self.transform == WeightTransform::WidenBf16F32;
        if widen && source.dtype != "BF16" {
            return refuse("only BF16 widens to F32");
        }
        if self.transform == WeightTransform::NarrowF32Bf16 && source.dtype != "F32" {
            return refuse("only F32 narrows to BF16");
        }
        if let WeightTransform::Fragment(format) = self.transform {
            let expected = match format {
                FragmentWeightFormat::Bf16 => ["BF16"].as_slice(),
                FragmentWeightFormat::Fp8E4M3 => ["F8_E4M3", "U8", "I8"].as_slice(),
            };
            if !expected.contains(&source.dtype.as_str()) {
                return refuse(format!(
                    "fragment format {format:?} cannot hold {} elements",
                    source.dtype
                ));
            }
        }
        if matches!(self.transform, WeightTransform::ScaleTile { .. })
            && source.element_bytes()? != 1
        {
            return refuse("scale tiles hold 1-byte scales");
        }
        if matches!(self.transform, WeightTransform::Mxfp4Moe(_)) && source.dtype != "U8" {
            return refuse("MXFP4 codes and scales are U8 bytes");
        }
        if self.transform == WeightTransform::CopyScales
            && !["F32", "BF16", "F8_E4M3"].contains(&source.dtype.as_str())
        {
            return refuse(format!(
                "checked scales are F32, BF16 or F8_E4M3 elements, not {}",
                source.dtype
            ));
        }
        gather::Gather::new(
            source,
            self.axis.map(|axis| axis as usize),
            &self.ranges,
            &self.stored_shape,
            widen,
        )
    }

    /// The fragment layout of the stored matrix: its last two dimensions,
    /// with every leading dimension 1.
    pub fn fragment_layout(
        &self,
        format: FragmentWeightFormat,
    ) -> Result<FragmentWeightLayout, WeightTransformError> {
        let (rows, k) = matrix(&self.stored_shape)?;
        let layout = FragmentWeightLayout { rows, k, format };
        layout.packed_bytes().map_err(refusal)?;
        Ok(layout)
    }

    /// The MXFP4 strip geometry of the stored `[rows, row_bytes]` matrix.
    pub fn mxfp4_strips(&self, section: Mxfp4Section) -> Result<Mxfp4Strips, WeightTransformError> {
        let (rows, row_bytes) = matrix(&self.stored_shape)?;
        let strips = Mxfp4Strips {
            section,
            rows,
            row_bytes,
        };
        strips.check()?;
        Ok(strips)
    }

    fn scale_tile_layout(
        &self,
        tile_rows: u32,
        k_blocks_per_tile: u32,
    ) -> Result<ScaleTileLayout, WeightTransformError> {
        let (rows, k_blocks) = matrix(&self.stored_shape)?;
        let (tile_rows, k_blocks_per_tile) = (tile_rows as usize, k_blocks_per_tile as usize);
        if tile_rows == 0 || k_blocks_per_tile == 0 || tile_rows * k_blocks_per_tile > 128 {
            return refuse("a scale tile holds 1 to 128 scales");
        }
        Ok(ScaleTileLayout {
            rows,
            k_blocks,
            tile_rows,
            k_blocks_per_tile,
        })
    }
}

/// `[rows, k]` of a stored shape whose leading dimensions are all 1.
fn matrix(shape: &[u64]) -> Result<(usize, usize), WeightTransformError> {
    let [lead @ .., rows, k] = shape else {
        return refuse("a matrix transform needs at least two stored dimensions");
    };
    if lead.iter().any(|dim| *dim != 1) {
        return refuse("a matrix transform stores one matrix");
    }
    Ok((
        usize::try_from(*rows).map_err(|_| WeightTransformError("rows overflow".into()))?,
        usize::try_from(*k).map_err(|_| WeightTransformError("k overflow".into()))?,
    ))
}

pub(crate) fn refusal(error: crate::contract::Refusal) -> WeightTransformError {
    WeightTransformError(error.0.into())
}

/// The BF16 nearest to an F32, ties to even; a NaN keeps its sign and upper
/// payload and becomes quiet. This is torch's conversion.
#[must_use]
pub const fn f32_to_bf16(bits: u32) -> u16 {
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}
