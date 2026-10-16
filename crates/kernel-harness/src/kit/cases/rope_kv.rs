//! Cases of the RoPE and KV write family (`kernel_core::rope_kv`): the RoPE
//! alone, the RoPE of packed QKV rows fused with the write of K and V into
//! a paged cache or a window ring (BF16 or FP8 E4M3), and the ring write;
//! the SM120 selection the lab's catalogs give, and the family's required
//! op set.
//!
//! Rows alternate between two items, each item's rows at consecutive
//! positions that cross a page boundary (paged) or wrap the ring (ring).
//! Block tables and ring slots are permuted ID inputs over the cache
//! planes, so the run also reruns them moved to the top of the planes.
//! Planes start from a nonzero image that every launch must keep outside
//! the slots its live rows write.
//!
//! The rotation's oracle is the contract's, `oracle::rope` in F64, and
//! each rotated element's bound follows from its own coefficients:
//!
//! - a table-backed rotation multiplies by the supplied F32 cosine and
//!   sine, so its bound is the F32 arithmetic of `x c -+ y s` (one product
//!   and one FMA, `2^-22 (|x c| + |y s|)`);
//! - a computed rotation's query does not pin how a backend gets each
//!   cosine and sine (SM120 rounds them to BF16, as the reference model
//!   does), so its bound adds `|x| dc + |y| ds`, where `dc` and `ds` are
//!   half a BF16 step of the exact cosine and sine plus the F32 angle's
//!   error (`ANGLE_ERROR` of the angle and of one radian).
//!
//! The gate admits the BF16 values or E4M3 codes that round the F32 values
//! within the bound (`conformance::admitted_range`). The case's tables hold angles hashed per
//! position and pair, not the computed ones, so a launch that ignores the
//! table fails. V is scaled by a power of two ([`VALUE_SCALE`]), so
//! `v_new`, the V plane and the ring write's copies and encodes compare as
//! exact bits; they give the ID reruns an exact result to tell moved IDs
//! apart.
//!
//! Ring planes, of the ring write and of a storing ring RoPE KV write, are
//! the planes `oracle::ring_kv_write` leaves ([`RingStore`]). Each computed
//! rotation also has an angle case ([`rope_angles`], [`rope_kv_angles`]):
//! its rotated pairs hold (1, 0), which a rotation turns into its cosine
//! and sine exactly before the BF16 output rounds them, and
//! `rotary::check_computed` judges each BF16-rounded cosine and sine. The
//! RoPE alone's angle case runs at positions up to 2^20 - 1.

use crate::LabResult;
use crate::conformance::{Expected, Holes, Images, Judge, Rows, Write};
use crate::kit::{OpCase, Required, Spec, bf16_bytes, bf16_values};
use crate::poison::WriteMap;
use kernel_core::backend::{Counts, Need, Prepared, View};
use kernel_core::codec::{SmallFloat, bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{DType, RotarySource, RotaryStyle, RowQuery};
use kernel_core::oracle::{self, OracleOutput, RingKvWrite};
use kernel_core::rope_kv::{
    KvCache, KvWrite, RingWriteBuffers, RingWriteOp, RopeBuffers, RopeKvBuffers, RopeKvOp, RopeOp,
};
use kernels_sm120::backend::RopeKvEntry;
use kernels_sm120::row::RowInstantiation;
use std::rc::Rc;

/// The relative error of a computed angle, `position / theta^(2i / d)` in
/// F32 arithmetic (a frequency from `exp2f` and `log2f` or a pinned table,
/// one product), and the absolute error of fast F32 trig after range
/// reduction, both bounded by 2^-20.
const ANGLE_ERROR: f64 = 1.0 / (1 << 20) as f64;

/// The F32 arithmetic of one rotated element, `2^-22` of its products'
/// magnitudes: one rounded product and one FMA.
const ROTATION_ARITHMETIC: f64 = 1.0 / (1 << 22) as f64;

/// Positions stay below this, as the cases' layouts keep them.
const MAX_POSITION: usize = 512;

/// The positions of the RoPE alone's angle case, by row: the first ones,
/// powers of two and their neighbours, and the largest below 2^20.
const ANGLE_POSITIONS: [usize; 16] = [
    0, 1, 2, 36, 511, 4095, 4096, 65535, 65536, 262143, 262144, 524287, 524288, 1_000_003,
    1_048_574, 1_048_575,
];

/// Half a BF16 step at `value`.
pub(crate) fn half_bf16(value: f64) -> f64 {
    if value == 0.0 {
        2_f64.powi(-134)
    } else {
        2_f64.powi((value.abs().log2().floor() as i32).max(-126) - 8)
    }
}

/// The multiplier of V the cases write: a power of two other than one, so a
/// launch that drops it fails and the product is exact. The rounding of
/// other multipliers is not exercised.
const VALUE_SCALE: f32 = 0.5;

/// The FP8 K and V tensor scales: distinct, so a launch that swaps or drops
/// one changes codes, and powers of two, so `value / scale` is exact.
const FP8_SCALES: [f32; 2] = [0.5, 0.25];

/// The values of BF16 `bits`.
fn widen(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_to_f32(b))).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Positions as `dtype` (I32 or I64) bytes.
fn position_bytes(positions: &[usize], dtype: DType) -> Vec<u8> {
    positions
        .iter()
        .flat_map(|&p| match dtype {
            DType::I64 => (p as i64).to_le_bytes().to_vec(),
            _ => (p as i32).to_le_bytes().to_vec(),
        })
        .collect()
}

/// The positions in `images` at `path`, I32 or I64.
fn positions_of(images: &Images, path: &str, dtype: DType) -> LabResult<Vec<usize>> {
    let bytes = images.bytes(path)?;
    Ok(match dtype {
        DType::I64 => bytes
            .chunks_exact(8)
            .map(|word| i64::from_le_bytes(word.try_into().expect("8 bytes")) as usize)
            .collect(),
        _ => images.i32s(path)?.into_iter().map(|p| p as usize).collect(),
    })
}

/// Each row's item and position as the images hold them (`token_batch`,
/// `positions` of `dtype`), so maps follow rows the property checks move.
fn placed_rows(images: &Images, dtype: DType) -> LabResult<Vec<(usize, usize)>> {
    let items = images.i32s("token_batch")?;
    let at = positions_of(images, "positions", dtype)?;
    Ok(items
        .iter()
        .zip(at)
        .map(|(&item, p)| (item as usize, p))
        .collect())
}

/// Each row's item and position over `capacity` rows: rows alternate
/// between the items (one item for a single row or a single sequence),
/// each item's rows at consecutive positions from its base.
fn placement(capacity: u32, bases: &[usize]) -> Vec<(usize, usize)> {
    let items = if capacity > 1 { bases.len() } else { 1 };
    (0..capacity as usize)
        .map(|row| (row % items, bases[row % items] + row / items))
        .collect()
}

/// A rotation: the query's math and, when table-backed, the case's F32
/// `[position][rotary / 2][cos, sin]` table.
#[derive(Clone, Copy)]
pub(crate) struct Rope {
    head_dim: usize,
    pub(crate) rotary: usize,
    theta: f64,
    style: RotaryStyle,
    pub(crate) table: bool,
}

impl Rope {
    pub(crate) fn of(query: RowQuery) -> LabResult<Self> {
        let RowQuery::Rope {
            head_dim,
            rotary_dim,
            theta,
            style,
            source,
            ..
        } = query
        else {
            return Err("a RoPE case rotates by a RoPE query".into());
        };
        Ok(Self {
            head_dim: head_dim as usize,
            rotary: rotary_dim as usize,
            theta: f64::from(theta),
            style,
            table: source == RotarySource::CosSinTable,
        })
    }

    /// The table of `positions` rows: per position and pair an angle hashed
    /// from both, unrelated to the computed angles, its cosine and sine
    /// rounded to F32.
    pub(crate) fn table_image(&self, positions: usize) -> Vec<u8> {
        let pairs = self.rotary / 2;
        (0..positions * pairs)
            .flat_map(|at| {
                let hash = kernel_core::ngram_oracle::splitmix64(0x0074_6162_6c65 ^ at as u64);
                let angle = (hash >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU;
                [angle.cos() as f32, angle.sin() as f32]
            })
            .flat_map(f32::to_le_bytes)
            .collect()
    }

    /// The pair of dimension `dim` of a head.
    fn pair(&self, dim: usize) -> usize {
        match self.style {
            RotaryStyle::Neox => dim % (self.rotary / 2),
            RotaryStyle::GptJ => dim / 2,
        }
    }

    /// The second dimension of the pair that rotary dimension `dim` leads,
    /// or `None` when `dim` is a pair's second or past the rotation.
    fn mate(&self, dim: usize) -> Option<usize> {
        let half = self.rotary / 2;
        match self.style {
            RotaryStyle::Neox => (dim < half).then_some(dim + half),
            RotaryStyle::GptJ => (dim < self.rotary && dim % 2 == 0).then_some(dim + 1),
        }
    }

    /// `bits`, rows of `stride` elements, with every rotated pair of the
    /// first `heads` heads of each row set to (1, 0). A rotation turns such
    /// a pair into its cosine and sine exactly: `1 * c - 0 * s` and
    /// `0 * c + 1 * s`.
    fn unit_pairs(&self, bits: &mut [u16], stride: usize, heads: usize) {
        let one = f32_to_bf16(1.0);
        for row in bits.chunks_exact_mut(stride) {
            for head in row[..heads * self.head_dim].chunks_exact_mut(self.head_dim) {
                for dim in 0..self.rotary {
                    if let Some(mate) = self.mate(dim) {
                        head[dim] = one;
                        head[mate] = 0;
                    }
                }
            }
        }
    }

    /// The judge of a computed rotation of unit pairs ([`Rope::unit_pairs`])
    /// into rows of `heads` heads: in each of the rows at `positions` (the
    /// live rows), every pair's cosine and sine within the allowance of
    /// `rotary::check_computed`, and every dimension past the rotation the
    /// bits of `input`, rows laid out as the output's.
    fn angle_judge(&self, input: Vec<u16>, heads: usize, positions: Vec<usize>) -> Judge {
        let rope = *self;
        Box::new(move |bytes: &[u8]| {
            let width = heads * rope.head_dim;
            let held = positions.len() * width;
            let output = bytes
                .get(..held * 2)
                .ok_or("the rotated rows are short")?
                .chunks_exact(2)
                .map(|half| u16::from_le_bytes([half[0], half[1]]))
                .collect::<Vec<_>>();
            let value = |bits: u16| f64::from(bf16_to_f32(bits));
            for (row, &position) in positions.iter().enumerate() {
                for head in 0..heads {
                    let at = row * width + head * rope.head_dim;
                    let (got, sent) = (
                        &output[at..][..rope.head_dim],
                        &input[at..][..rope.head_dim],
                    );
                    if got[rope.rotary..] != sent[rope.rotary..] {
                        return Err(format!(
                            "row {row} head {head}: a dimension past the rotation changed"
                        ));
                    }
                    for dim in 0..rope.rotary {
                        let Some(mate) = rope.mate(dim) else {
                            continue;
                        };
                        let pair = rope.pair(dim);
                        let cos_sin = (value(got[dim]), value(got[mate]));
                        kernel_core::rotary::check_computed(
                            rope.theta,
                            rope.rotary,
                            pair,
                            position,
                            cos_sin,
                        )
                        .map_err(|refusal| {
                            format!(
                                "row {row} head {head} pair {pair} at position {position}: \
                                 {cos_sin:?}: {}",
                                refusal.0
                            )
                        })?;
                    }
                }
            }
            Ok(())
        })
    }

    /// The exact cosine and sine of `pair` at `position`, and their error
    /// beyond the F32 arithmetic: none from a table, a BF16 rounding plus
    /// the angle's error when computed.
    fn coefficients(
        &self,
        table: Option<&[(f64, f64)]>,
        position: usize,
        pair: usize,
    ) -> ((f64, f64), (f64, f64)) {
        if let Some(table) = table {
            return (table[position * (self.rotary / 2) + pair], (0.0, 0.0));
        }
        let angle = position as f64 / self.theta.powf(2.0 * pair as f64 / self.rotary as f64);
        let error = ANGLE_ERROR * (angle.abs() + 1.0);
        let (c, s) = (angle.cos(), angle.sin());
        ((c, s), (half_bf16(c) + error, half_bf16(s) + error))
    }

    /// The rotation of `x`, rows of `heads` heads at `positions`, by the
    /// table at role `table` of `images` when table-backed, and each
    /// element's allowance beyond its output's rounding.
    pub(crate) fn apply(
        &self,
        images: &Images,
        table: &str,
        x: &[f64],
        heads: usize,
        positions: &[usize],
    ) -> LabResult<(Vec<f64>, Vec<f64>)> {
        let expanded = positions
            .iter()
            .flat_map(|&p| std::iter::repeat_n(p, heads))
            .collect::<Vec<_>>();
        if expanded.iter().any(|&p| p >= MAX_POSITION) {
            return Err(format!("RoPE cases keep positions below {MAX_POSITION}"));
        }
        let table = if self.table {
            Some(
                images
                    .f32s(table)?
                    .chunks_exact(2)
                    .map(|pair| (f64::from(pair[0]), f64::from(pair[1])))
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let values = oracle::rope(
            x,
            &expanded,
            self.head_dim,
            self.rotary,
            self.theta,
            self.style,
            table.as_deref(),
        )
        .map_err(|e| e.0.to_string())?;
        let half = self.rotary / 2;
        let allowance = (0..x.len())
            .map(|at| {
                let dim = at % self.head_dim;
                if dim >= self.rotary {
                    return 0.0;
                }
                let mate = match self.style {
                    RotaryStyle::Neox if dim < half => at + half,
                    RotaryStyle::Neox => at - half,
                    RotaryStyle::GptJ => at ^ 1,
                };
                let position = expanded[at / self.head_dim];
                let ((c, s), (dc, ds)) =
                    self.coefficients(table.as_deref(), position, self.pair(dim));
                let (x, y) = (x[at].abs(), x[mate].abs());
                x * dc + y * ds + ROTATION_ARITHMETIC * (x * c.abs() + y * s.abs())
            })
            .collect();
        Ok((values, allowance))
    }
}

/// An oracle output of `values` within each element's `allowance` (as
/// 2^-22 K S with K = 1), which the gate compares as the BF16 values or
/// E4M3 codes that round the allowance's ends.
fn bounded(values: Vec<f64>, allowance: &[f64]) -> LabResult<Expected> {
    let sums = allowance.iter().map(|a| a * 2_f64.powi(22)).collect();
    Ok(Expected::Values(
        OracleOutput::new(values, sums, 1).map_err(|e| e.0.to_string())?,
    ))
}

/// The bits of BF16 `v` times the op's power-of-two multiplier, exact.
fn scaled(v: &[u16], scale: f32) -> LabResult<Vec<u16>> {
    v.iter()
        .map(|&bits| {
            let value = bf16_to_f32(bits) * scale;
            let scaled = kernel_core::codec::f32_to_bf16(value);
            if bf16_to_f32(scaled) == value {
                Ok(scaled)
            } else {
                Err(format!("V times {scale} is not a BF16 value"))
            }
        })
        .collect()
}

/// A cache plane's values from its initial `image` (BF16 values, or FP8
/// codes in their units), with no allowance.
fn plane_values(image: &[u8], dtype: DType) -> (Vec<f64>, Vec<f64>) {
    let values = match dtype {
        DType::Fp8E4M3 => image
            .iter()
            .map(|&code| SmallFloat::E4M3.decode_f64(code))
            .collect::<Vec<_>>(),
        _ => widen(
            &image
                .chunks_exact(2)
                .map(|half| u16::from_le_bytes([half[0], half[1]]))
                .collect::<Vec<_>>(),
        ),
    };
    let allowance = vec![0.0; values.len()];
    (values, allowance)
}

/// An initial plane image of `elements`: BF16 values from the kit's
/// generator, or E4M3 codes of every finite value.
fn plane_image(elements: usize, dtype: DType, seed: u64) -> Vec<u8> {
    match dtype {
        DType::Fp8E4M3 => (0..elements)
            .map(|at| {
                let code = (at.wrapping_mul(37) + seed as usize) as u8;
                if code & 0x7f == 0x7f { code - 1 } else { code }
            })
            .collect(),
        _ => bf16_bytes(&bf16_values(seed, elements)),
    }
}

fn kv_bytes(dtype: DType) -> usize {
    if dtype == DType::Fp8E4M3 { 1 } else { 2 }
}

/// Where a cache write stores K and V: slots of `tokens` tokens of
/// `kv_heads` heads.
#[derive(Clone, Copy)]
struct Planes {
    dtype: DType,
    kv_heads: usize,
    /// The tokens of a page or a ring.
    tokens: usize,
    slots: usize,
    paged: bool,
    /// The FP8 K and V scales, one for BF16.
    scales: [f32; 2],
}

impl Planes {
    /// Elements of one slot of the plane of head width `width`.
    fn slot(&self, width: usize) -> usize {
        self.tokens * self.kv_heads * width
    }

    /// The slot and token of each live row's write.
    fn targets(
        &self,
        images: &Images,
        live: u32,
        rows: &[(usize, usize)],
        index_path: &str,
        pages: usize,
    ) -> LabResult<Vec<(usize, usize, usize)>> {
        let index = images.i32s(index_path)?;
        rows.iter()
            .take(live as usize)
            .enumerate()
            .map(|(row, &(item, position))| {
                let slot = if self.paged {
                    index[item * pages + position / self.tokens]
                } else {
                    index[item]
                };
                let slot = usize::try_from(slot).map_err(|_| "a hole in a cache index")?;
                Ok((row, slot, position % self.tokens))
            })
            .collect()
    }

    /// The bytes of `width`-wide plane elements the targets write.
    fn write_map(&self, targets: &[(usize, usize, usize)], width: usize) -> LabResult<WriteMap> {
        let row = self.kv_heads * width;
        WriteMap::of(self.dtype).ranges(targets.iter().map(|&(_, slot, token)| {
            let start = slot * self.slot(width) + token * row;
            start..start + row
        }))
    }

    /// The plane of `width` after the targets write `values` (rows of
    /// `kv_heads * width`, with their allowance) over its initial image, in
    /// the plane's units: an FP8 plane holds codes of `value / scale`.
    fn after(
        &self,
        image: &[u8],
        targets: &[(usize, usize, usize)],
        width: usize,
        (values, allowance): (&[f64], &[f64]),
        scale: f32,
    ) -> LabResult<Expected> {
        let (mut plane, mut bound) = plane_values(image, self.dtype);
        let row = self.kv_heads * width;
        let divisor = if self.dtype == DType::Fp8E4M3 {
            f64::from(scale)
        } else {
            1.0
        };
        for &(at, slot, token) in targets {
            let start = slot * self.slot(width) + token * row;
            for element in 0..row {
                plane[start + element] = values[at * row + element] / divisor;
                bound[start + element] = allowance[at * row + element] / divisor;
            }
        }
        bounded(plane, &bound)
    }

    /// The plane of `width` after the targets write the BF16 `rows` (rows
    /// of `kv_heads * width`) over its initial image: their bits, or the
    /// E4M3 codes of `value / scale`.
    fn after_bits(
        &self,
        image: &[u8],
        targets: &[(usize, usize, usize)],
        width: usize,
        rows: &[u16],
        scale: f32,
    ) -> Vec<u8> {
        let mut plane = image.to_vec();
        let row = self.kv_heads * width;
        for &(at, slot, token) in targets {
            let start = slot * self.slot(width) + token * row;
            let values = &rows[at * row..][..row];
            if self.dtype == DType::Fp8E4M3 {
                for (element, &bits) in values.iter().enumerate() {
                    plane[start + element] = kernel_core::kv_fp8::encode(bf16_to_f32(bits), scale);
                }
            } else {
                plane[start * 2..][..row * 2].copy_from_slice(&bf16_bytes(values));
            }
        }
        plane
    }

    /// The plane of `image` with each element of `stored` that is not NaN
    /// in place: its BF16 bits, or the E4M3 code of `value / scale`. The
    /// values are BF16 rows, so both are exact.
    fn stored_bits(&self, image: &[u8], stored: &[f64], scale: f32) -> LabResult<Vec<u8>> {
        let mut plane = image.to_vec();
        for (at, &value) in stored.iter().enumerate().filter(|(_, v)| !v.is_nan()) {
            let bits = f32_to_bf16(value as f32);
            if f64::from(bf16_to_f32(bits)) != value {
                return Err(format!("a ring row's {value} is not a BF16 value"));
            }
            if self.dtype == DType::Fp8E4M3 {
                plane[at] = kernel_core::kv_fp8::encode(value as f32, scale);
            } else {
                plane[at * 2..][..2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        Ok(plane)
    }

    /// The plane of `image` with each element of `stored` that is not NaN
    /// in place within its `allowance`, both in the plane's units.
    fn stored_values(
        &self,
        image: &[u8],
        stored: &[f64],
        allowance: &[f64],
    ) -> LabResult<Expected> {
        let (mut plane, mut bound) = plane_values(image, self.dtype);
        for (at, &value) in stored.iter().enumerate().filter(|(_, v)| !v.is_nan()) {
            plane[at] = value;
            bound[at] = allowance[at];
        }
        bounded(plane, &bound)
    }
}

/// A ring store as `oracle::ring_kv_write` states it, over the live rows:
/// each row's item and position, each item's ring and last position. A row
/// a ring or more before its item's last position is superseded and not
/// stored.
struct RingStore {
    planes: Planes,
    /// The K and V head widths.
    widths: [usize; 2],
    token_batch: Vec<usize>,
    positions: Vec<usize>,
    slot_ids: Vec<usize>,
    last_positions: Vec<usize>,
}

impl RingStore {
    /// The store of the first `live` of `rows` (item, position) into the
    /// rings `slot_ids` (an image's I32 IDs) names.
    fn new(
        planes: Planes,
        widths: [usize; 2],
        rows: &[(usize, usize)],
        live: u32,
        slot_ids: &[i32],
        last_positions: Vec<usize>,
    ) -> LabResult<Self> {
        let slot_ids = slot_ids
            .iter()
            .map(|&slot| usize::try_from(slot).map_err(|_| "a hole in a ring's slot IDs"))
            .collect::<Result<_, _>>()?;
        let rows = &rows[..live as usize];
        Ok(Self {
            planes,
            widths,
            token_batch: rows.iter().map(|&(item, _)| item).collect(),
            positions: rows.iter().map(|&(_, p)| p).collect(),
            slot_ids,
            last_positions,
        })
    }

    /// The K and V planes after the oracle writes the live rows of `k` and
    /// `v` (rows of `kv_heads` heads of each width) into planes of NaN, so
    /// NaN marks every element it leaves. Refuses non-finite rows, which
    /// would read as left.
    fn store(&self, k: &[f64], v: &[f64]) -> LabResult<[Vec<f64>; 2]> {
        let [qk, vd] = self.widths;
        let (kvh, rows) = (self.planes.kv_heads, self.positions.len());
        let (k, v) = (&k[..rows * kvh * qk], &v[..rows * kvh * vd]);
        if k.iter().chain(v).any(|x| !x.is_finite()) {
            return Err("a ring write's live rows hold a non-finite value".into());
        }
        let plane = |width: usize| vec![f64::NAN; self.planes.slots * self.planes.slot(width)];
        let (mut k_ring, mut v_ring) = (plane(qk), plane(vd));
        oracle::ring_kv_write(RingKvWrite {
            k_ring: &mut k_ring,
            v_ring: &mut v_ring,
            k_new: k,
            v_new: v,
            positions: &self.positions,
            token_batch: &self.token_batch,
            slot_ids: &self.slot_ids,
            last_positions: &self.last_positions,
            ring: self.planes.tokens,
            kv_heads: kvh,
            qk_dim: qk,
            v_dim: vd,
        })
        .map_err(|e| e.0.to_string())?;
        Ok([k_ring, v_ring])
    }
}

/// The case of a RoPE KV write over `capacity` rows: packed BF16 QKV rows
/// of two items (bases chosen so paged rows cross a page and ring rows
/// wrap the ring), each item's block table row or ring a permuted ID
/// input, and planes of nonzero images; the oracle rotates Q and K, scales
/// V and stores K and V where the op says, a ring's K and V as
/// [`RingStore`]. With `angles` the Q and K heads' rotated pairs are unit
/// pairs and `q` and `k_new` are judged by [`Rope::angle_judge`].
pub fn rope_kv_case<'a>(
    op: RopeKvOp,
    capacity: u32,
    prepared: Prepared<RopeKvBuffers<View>>,
    angles: bool,
) -> LabResult<OpCase<'a>> {
    let rope = Rope::of(op.rope)?;
    let (qh, kvh, qk, v) = (
        op.q_heads as usize,
        op.kv_heads as usize,
        rope.head_dim,
        op.v_dim as usize,
    );
    let stride = qh * qk + kvh * (qk + v);
    let rows = capacity as usize;
    let (tokens, store, pages, kv_dtype) = match op.cache {
        KvWrite::Paged {
            page_size,
            max_context,
            kv_dtype,
        } => (
            page_size as usize,
            true,
            max_context.div_ceil(page_size) as usize,
            kv_dtype,
        ),
        KvWrite::Ring {
            ring_size,
            kv_dtype,
            store,
        } => (ring_size as usize, store, 1, kv_dtype),
    };
    let paged = matches!(op.cache, KvWrite::Paged { .. });
    let bases = if paged {
        [tokens.saturating_sub(3), 2 * tokens + 1]
    } else {
        [tokens - 4, 3]
    };
    let placed = placement(capacity, &bases);
    let items = if capacity > 1 { 2 } else { 1 };
    let last = placed.iter().map(|&(_, p)| p).max().unwrap_or(0);
    if let KvWrite::Paged { max_context, .. } = op.cache
        && last >= max_context as usize
    {
        return Err(format!(
            "the case writes position {last}, past the op's context {max_context}"
        ));
    }
    let table_rows = last + 1;
    // Paged: each item's pages, a permutation of the slots below the top
    // one. Ring: each item's ring, reversed.
    let slots = items * pages + 1;
    let index = (0..items * pages)
        .map(|at| (items * pages - 1 - at) as i32)
        .collect::<Vec<_>>();
    let planes = Planes {
        dtype: kv_dtype,
        kv_heads: kvh,
        tokens,
        slots,
        paged,
        scales: if kv_dtype == DType::Fp8E4M3 {
            FP8_SCALES
        } else {
            [1.0; 2]
        },
    };
    let eb = kv_bytes(kv_dtype);
    let index_path = if paged { "block_tables" } else { "slot_ids" };
    let pools = [
        ("cache.k", planes.slot(qk) * eb),
        ("cache.v", planes.slot(v) * eb),
    ];
    let index = Spec::ids(
        (Rows::Fixed(items), pages),
        index,
        if store { &pools } else { &[] },
        Holes::Never,
    );
    let (block_tables, slot_ids) = if paged {
        (Some(index), None)
    } else {
        (None, Some(index))
    };
    let plane = |width: usize, seed: u64, which: usize| -> Spec<'a> {
        let elements = slots * planes.slot(width);
        if !store {
            return Spec::untouched(elements * eb);
        }
        Spec::state(
            kv_dtype,
            (Rows::Fixed(slots), planes.slot(width)),
            plane_image(elements, kv_dtype, seed),
            Write::Map(Box::new(move |images, live| {
                let rows = placed_rows(images, op.positions)?;
                let targets = planes.targets(images, live, &rows, index_path, pages)?;
                planes.write_map(&targets, [qk, v][which])
            })),
        )
    };
    let mut qkv = bf16_values(11, rows * stride);
    if angles {
        rope.unit_pairs(&mut qkv, stride, qh + kvh);
    }
    let roles = RopeKvBuffers {
        qkv: Spec::input(DType::Bf16, (Rows::Live, stride), bf16_bytes(&qkv)),
        q: Spec::output(DType::Bf16, (Rows::Live, qh * qk)),
        k_new: Some(Spec::output(DType::Bf16, (Rows::Live, kvh * qk))),
        v_new: Some(Spec::output(DType::Bf16, (Rows::Live, kvh * v))),
        cache: KvCache {
            k: plane(qk, 12, 0),
            v: plane(v, 13, 1),
            slots: slots as u32,
            k_slot_stride: (planes.slot(qk) * eb) as u64,
            v_slot_stride: (planes.slot(v) * eb) as u64,
            k_scale: planes.scales[0],
            v_scale: planes.scales[1],
        },
        block_tables,
        slot_ids,
        token_batch: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(
                &placed
                    .iter()
                    .map(|&(item, _)| item as i32)
                    .collect::<Vec<_>>(),
            ),
        ),
        positions: Spec::input(
            op.positions,
            (Rows::Live, 1),
            position_bytes(
                &placed.iter().map(|&(_, p)| p).collect::<Vec<_>>(),
                op.positions,
            ),
        ),
        cos_sin: rope.table.then(|| {
            Spec::input(
                DType::F32,
                (Rows::Fixed(table_rows), rope.rotary),
                rope.table_image(table_rows),
            )
        }),
        live_rows: Spec::live(),
    };
    let value_scale = op.value_scale;
    let positions = op.positions;
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let bits = images.u16s("qkv")?;
        let at = positions_of(images, "positions", positions)?;
        let rows = placed_rows(images, positions)?;
        let part = |from: usize, width: usize| {
            bits.chunks_exact(stride)
                .flat_map(|row| row[from..from + width].to_vec())
                .collect::<Vec<_>>()
        };
        let (q_bits, k_bits) = (part(0, qh * qk), part(qh * qk, kvh * qk));
        let (q, q_bound) = rope.apply(images, "cos_sin", &widen(&q_bits), qh, &at)?;
        let (k, k_bound) = rope.apply(images, "cos_sin", &widen(&k_bits), kvh, &at)?;
        let vs = scaled(&part(qh * qk + kvh * qk, kvh * v), value_scale)?;
        let [q_expected, k_expected] = if angles {
            let live_at = at[..live as usize].to_vec();
            [
                Expected::Judge(rope.angle_judge(q_bits, qh, live_at.clone())),
                Expected::Judge(rope.angle_judge(k_bits, kvh, live_at)),
            ]
        } else {
            [bounded(q, &q_bound)?, bounded(k.clone(), &k_bound)?]
        };
        let mut results = vec![
            ("q".to_owned(), q_expected),
            ("k_new".to_owned(), k_expected),
            ("v_new".to_owned(), Expected::Bits(bf16_bytes(&vs))),
        ];
        if store && paged {
            let targets = planes.targets(images, live, &rows, index_path, pages)?;
            results.push((
                "cache.k".to_owned(),
                planes.after(
                    images.bytes("cache.k")?,
                    &targets,
                    qk,
                    (&k, &k_bound),
                    planes.scales[0],
                )?,
            ));
            results.push((
                "cache.v".to_owned(),
                Expected::Bits(planes.after_bits(
                    images.bytes("cache.v")?,
                    &targets,
                    v,
                    &vs,
                    planes.scales[1],
                )),
            ));
        } else if store {
            // The fused write stores every live row: as the ring write
            // with each item's last position its last live row's, while no
            // item's live rows span a ring.
            let slot_ids = images.i32s("slot_ids")?;
            let mut last_positions = vec![0; slot_ids.len()];
            for &(item, position) in &rows[..live as usize] {
                let last = last_positions
                    .get_mut(item)
                    .ok_or("a row's item has no ring")?;
                *last = (*last).max(position);
            }
            if rows[..live as usize]
                .iter()
                .any(|&(item, position)| position + tokens <= last_positions[item])
            {
                return Err("an item's live rows span its ring".into());
            }
            let ring = RingStore::new(planes, [qk, v], &rows, live, &slot_ids, last_positions)?;
            // K in the plane's units, an FP8 plane's `value / scale`.
            let divisor = if kv_dtype == DType::Fp8E4M3 {
                f64::from(planes.scales[0])
            } else {
                1.0
            };
            let units = |rows: &[f64]| rows.iter().map(|x| x / divisor).collect::<Vec<_>>();
            let vs = widen(&vs);
            let [k_plane, v_plane] = ring.store(&units(&k), &vs)?;
            let [k_allowance, _] = ring.store(&units(&k_bound), &vs)?;
            results.push((
                "cache.k".to_owned(),
                planes.stored_values(images.bytes("cache.k")?, &k_plane, &k_allowance)?,
            ));
            results.push((
                "cache.v".to_owned(),
                Expected::Bits(planes.stored_bits(
                    images.bytes("cache.v")?,
                    &v_plane,
                    planes.scales[1],
                )?),
            ));
        }
        Ok(results)
    };
    let positions = table_rows as u32;
    Ok(OpCase::with_counts(prepared, capacity, roles, move |rows| {
        Counts::rows(rows).with_positions(positions)
    })?
    .oracle(oracle))
}

/// The case of a RoPE alone over `capacity` rows of `heads` BF16 heads:
/// one sequence from position 5, and the rotation's oracle. With `angles`
/// the rows hold unit pairs at [`ANGLE_POSITIONS`] and the output is
/// judged by [`Rope::angle_judge`].
pub fn rope_case<'a>(
    op: RopeOp,
    capacity: u32,
    prepared: Prepared<RopeBuffers<View>>,
    angles: bool,
) -> LabResult<OpCase<'a>> {
    let rope = Rope::of(op.rope)?;
    if angles && rope.table {
        return Err("an angle case rotates by computed angles".into());
    }
    let width = op.heads as usize * rope.head_dim;
    let placed = (0..capacity as usize)
        .map(|row| {
            if angles {
                ANGLE_POSITIONS[row % ANGLE_POSITIONS.len()]
            } else {
                5 + row
            }
        })
        .collect::<Vec<_>>();
    let table_rows = placed.iter().copied().max().unwrap_or(0) + 1;
    let mut input = bf16_values(14, capacity as usize * width);
    if angles {
        rope.unit_pairs(&mut input, width, op.heads as usize);
    }
    let roles = RopeBuffers {
        input: Spec::input(DType::Bf16, (Rows::Live, width), bf16_bytes(&input)),
        output: Spec::output(DType::Bf16, (Rows::Live, width)),
        positions: Spec::input(
            op.positions,
            (Rows::Live, 1),
            position_bytes(&placed, op.positions),
        ),
        cos_sin: rope.table.then(|| {
            Spec::input(
                DType::F32,
                (Rows::Fixed(table_rows), rope.rotary),
                rope.table_image(table_rows),
            )
        }),
        live_rows: Spec::live(),
    };
    let (heads, positions) = (op.heads as usize, op.positions);
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let at = positions_of(images, "positions", positions)?;
        let input = images.u16s("input")?;
        if angles {
            let live_at = at[..live as usize].to_vec();
            let judge = rope.angle_judge(input, heads, live_at);
            return Ok(vec![("output".to_owned(), Expected::Judge(judge))]);
        }
        let (values, bound) = rope.apply(images, "cos_sin", &widen(&input), heads, &at)?;
        Ok(vec![("output".to_owned(), bounded(values, &bound)?)])
    };
    let positions = table_rows as u32;
    Ok(OpCase::with_counts(prepared, capacity, roles, move |rows| {
        Counts::rows(rows).with_positions(positions)
    })?
    .oracle(oracle))
}

/// The case of a ring write over `capacity` rows: BF16 K and V rows of two
/// items (one for a single sequence), the first wrapping the ring and the
/// second's first rows superseded by a later row (`last_positions`), each
/// item's ring a permuted ID input; the planes [`RingStore`] gives, each
/// stored row's copy (BF16) or E4M3 encode of `value / scale` (FP8), as
/// exact bits.
pub fn ring_write_case<'a>(
    op: RingWriteOp,
    capacity: u32,
    prepared: Prepared<RingWriteBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let RowQuery::RingWrite {
        qk_dim,
        v_dim,
        kv_heads,
        ring_size,
        kv_dtype,
    } = op.query
    else {
        return Err("a ring write case writes by a ring write query".into());
    };
    let (qk, v, kvh, ring) = (
        qk_dim as usize,
        v_dim as usize,
        kv_heads as usize,
        ring_size as usize,
    );
    let bases: &[usize] = if op.single_sequence {
        &[ring - 4]
    } else {
        &[ring - 4, 3]
    };
    let placed = placement(capacity, bases);
    let items = if capacity > 1 { bases.len() } else { 1 };
    // Item 0's last row is its own; item 1's would come a ring and two rows
    // after its first, so the rows at its first three positions are
    // superseded.
    let last_positions = (0..items)
        .map(|item| {
            let own = placed
                .iter()
                .filter(|&&(i, _)| i == item)
                .map(|&(_, p)| p)
                .max()
                .unwrap_or(0);
            if item == 1 { bases[1] + ring + 2 } else { own }
        })
        .collect::<Vec<_>>();
    let planes = Planes {
        dtype: kv_dtype,
        kv_heads: kvh,
        tokens: ring,
        slots: items + 1,
        paged: false,
        scales: if kv_dtype == DType::Fp8E4M3 {
            FP8_SCALES
        } else {
            [1.0; 2]
        },
    };
    let eb = kv_bytes(kv_dtype);
    // The live rows a launch stores (row, ring, token), superseded rows
    // dropped, by the items, positions and slots in the images.
    let position_type = op.positions;
    let stored = Rc::new(
        move |images: &Images, live: u32| -> LabResult<Vec<(usize, usize, usize)>> {
            let slots = images.i32s("slot_ids")?;
            let last = positions_of(images, "last_positions", DType::I64)?;
            Ok(placed_rows(images, position_type)?
                .into_iter()
                .take(live as usize)
                .enumerate()
                .filter(|&(_, (item, p))| p + ring > last[item])
                .map(|(row, (item, p))| (row, slots[item] as usize, p % ring))
                .collect())
        },
    );
    let plane = |width: usize, seed: u64| -> Spec<'a> {
        let stored = Rc::clone(&stored);
        Spec::state(
            kv_dtype,
            (Rows::Fixed(planes.slots), planes.slot(width)),
            plane_image(planes.slots * planes.slot(width), kv_dtype, seed),
            Write::Map(Box::new(move |images, live| {
                planes.write_map(&stored(images, live)?, width)
            })),
        )
    };
    let pools = [
        ("cache.k", planes.slot(qk) * eb),
        ("cache.v", planes.slot(v) * eb),
    ];
    let rows = capacity as usize;
    let roles = RingWriteBuffers {
        k_new: Spec::input(
            DType::Bf16,
            (Rows::Live, kvh * qk),
            bf16_bytes(&bf16_values(15, rows * kvh * qk)),
        ),
        v_new: Spec::input(
            DType::Bf16,
            (Rows::Live, kvh * v),
            bf16_bytes(&bf16_values(16, rows * kvh * v)),
        ),
        cache: KvCache {
            k: plane(qk, 17),
            v: plane(v, 18),
            slots: planes.slots as u32,
            k_slot_stride: (planes.slot(qk) * eb) as u64,
            v_slot_stride: (planes.slot(v) * eb) as u64,
            k_scale: planes.scales[0],
            v_scale: planes.scales[1],
        },
        slot_ids: Spec::ids(
            (Rows::Fixed(items), 1),
            (0..items).rev().map(|item| item as i32).collect(),
            &pools,
            Holes::Never,
        ),
        token_batch: Spec::input(
            DType::I32,
            (Rows::Live, 1),
            i32_bytes(&placed.iter().map(|&(i, _)| i as i32).collect::<Vec<_>>()),
        ),
        positions: Spec::input(
            op.positions,
            (Rows::Live, 1),
            position_bytes(
                &placed.iter().map(|&(_, p)| p).collect::<Vec<_>>(),
                op.positions,
            ),
        ),
        last_positions: Spec::input(
            DType::I64,
            (Rows::Fixed(items), 1),
            position_bytes(&last_positions, DType::I64),
        ),
        live_rows: Spec::live(),
    };
    // The planes `oracle::ring_kv_write` leaves, apart from the write maps'
    // reading of the same images.
    let oracle = move |images: &Images, live: u32| -> LabResult<Vec<(String, Expected)>> {
        let ring = RingStore::new(
            planes,
            [qk, v],
            &placed_rows(images, position_type)?,
            live,
            &images.i32s("slot_ids")?,
            positions_of(images, "last_positions", DType::I64)?,
        )?;
        let stored = ring.store(
            &widen(&images.u16s("k_new")?),
            &widen(&images.u16s("v_new")?),
        )?;
        let mut results = Vec::new();
        for ((plane, scale), stored) in ["cache.k", "cache.v"]
            .into_iter()
            .zip(planes.scales)
            .zip(stored)
        {
            let bits = planes.stored_bits(images.bytes(plane)?, &stored, scale)?;
            results.push((plane.to_owned(), Expected::Bits(bits)));
        }
        Ok(results)
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// The required RoPE KV write `op` over `need`.
pub fn rope_kv<'a>(name: String, op: RopeKvOp, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.rope_kv_write(&op, need),
        move |prepared| rope_kv_case(op, need.capacity, prepared, false),
    )
}

/// The required RoPE KV write `op` over `need`, a computed rotation, with
/// its rotated pairs judged by `rotary::check_computed`.
pub fn rope_kv_angles<'a>(name: String, op: RopeKvOp, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.rope_kv_write(&op, need),
        move |prepared| rope_kv_case(op, need.capacity, prepared, true),
    )
}

/// The required RoPE alone `op` over `need`.
pub fn rope<'a>(name: String, op: RopeOp, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.rope(&op, need),
        move |prepared| rope_case(op, need.capacity, prepared, false),
    )
}

/// The required RoPE alone `op` over `need`, a computed rotation, with its
/// rotated pairs judged by `rotary::check_computed` up to position
/// 2^20 - 1.
pub fn rope_angles<'a>(name: String, op: RopeOp, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.rope(&op, need),
        move |prepared| rope_case(op, need.capacity, prepared, true),
    )
}

/// The required ring write `op` over `need`.
pub fn ring_write<'a>(name: String, op: RingWriteOp, need: Need) -> Required<'a> {
    Required::new(
        name,
        need,
        move |backend, need| backend.ring_write(&op, need),
        move |prepared| ring_write_case(op, need.capacity, prepared),
    )
}

/// The template arguments of an entry body `chop::row::NAME<...>(`, if it
/// instantiates `name`.
fn template<'b>(body: &'b str, name: &str) -> Option<Vec<&'b str>> {
    let rest = body.strip_prefix(&format!("chop::row::{name}<"))?;
    let (arguments, _) = rest.split_once(">(")?;
    Some(arguments.split(',').collect())
}

/// The family's entries of `catalogs`, the first of each launch:
/// equal instantiations under two symbols are an SM120 tie, and a ring
/// RoPE KV write ignores its page size, so ring entries of two page sizes
/// are one.
pub fn sm120_entries(catalogs: &kernels_sm120::Catalogs) -> Vec<RopeKvEntry> {
    let mut chosen: Vec<(String, RowInstantiation)> = Vec::new();
    for row in kernels_sm120::row::catalog(catalogs) {
        let key = if let Some(arguments) = template(&row.body, "rope_kv") {
            // A ring (the eighth argument) launches alike at any page size.
            let ring = arguments.get(7) == Some(&"true");
            let shown = if ring {
                &arguments[..arguments.len() - 1]
            } else {
                &arguments[..]
            };
            format!("rope_kv {:?} {}", row.query, shown.join(","))
        } else if template(&row.body, "ring_write").is_some()
            || template(&row.body, "dflash_rope").is_some()
        {
            format!("{:?} {} {:?}", row.query, row.body, row.grid)
        } else {
            continue;
        };
        if !chosen.iter().any(|(other, _)| *other == key) {
            chosen.push((key, row));
        }
    }
    chosen
        .into_iter()
        .map(|(_, row)| RopeKvEntry::new(row))
        .collect()
}

/// The positions type of a template's `Pos64` argument.
fn positions(pos64: &str) -> DType {
    if pos64 == "true" {
        DType::I64
    } else {
        DType::I32
    }
}

/// The ring every ring op of the family writes: the catalogs' ring write
/// size.
fn ring_size(entries: &[RopeKvEntry]) -> Option<u32> {
    entries.iter().find_map(|entry| match entry.entry.query {
        RowQuery::RingWrite { ring_size, .. } => Some(ring_size),
        _ => None,
    })
}

/// The context of a paged op over `capacity` rows of [`rope_kv_case`]'s
/// placement at page size `page`: whole pages past its last position.
fn paged_context(page: u32, capacity: u32) -> u32 {
    let last = 2 * page + 1 + capacity.div_ceil(2);
    (last + 1).div_ceil(page) * page
}

/// The rows the family's ops serve, and the larger capacity one paged op
/// of each KV dtype runs at, where SM120 blocks rotate four Q heads.
pub const CAPACITY: u32 = super::ROW_CAPACITY;
pub const WIDE_CAPACITY: u32 = 512;

/// The family's required ops, one per SM120 catalog entry `entries` holds,
/// as a model states it: a RoPE KV write per `rope_kv` entry (its heads,
/// widths, cache and positions type; paged ops over a context of whole
/// pages past the case's positions, ring ops over the catalogs' ring), a
/// RoPE alone per `dflash_rope` entry and a ring write per `ring_write`
/// entry, each over [`CAPACITY`] rows; the first paged op of each KV
/// dtype again over [`WIDE_CAPACITY`] rows; and each computed rotation's
/// angle op (named `... angles`) over [`CAPACITY`] rows.
pub fn family<'a>(entries: &[RopeKvEntry]) -> LabResult<Vec<Required<'a>>> {
    let ring = ring_size(entries);
    let mut ops = Vec::new();
    let mut wide = Vec::new();
    for RopeKvEntry { entry, .. } in entries {
        if let Some(arguments) = template(&entry.body, "rope_kv") {
            let number = |at: usize| -> LabResult<u32> {
                arguments[at]
                    .parse()
                    .map_err(|_| format!("{}: template argument {at}", entry.symbol))
            };
            let flag = |at: usize| arguments[at] == "true";
            let kv_dtype = if flag(9) { DType::Fp8E4M3 } else { DType::Bf16 };
            let cache = if flag(7) {
                KvWrite::Ring {
                    ring_size: ring.ok_or("a ring RoPE KV write without a ring write")?,
                    kv_dtype,
                    store: !flag(8),
                }
            } else {
                let page_size = number(11)?;
                KvWrite::Paged {
                    page_size,
                    max_context: paged_context(page_size, WIDE_CAPACITY),
                    kv_dtype,
                }
            };
            let op = RopeKvOp {
                rope: entry.query,
                q_heads: number(0)?,
                kv_heads: number(1)?,
                v_dim: number(3)?,
                value_scale: VALUE_SCALE,
                positions: positions(arguments[10]),
                cache,
            };
            let name = |capacity| {
                format!(
                    "rope kv {:?} q{} kv{} v{} {} {:?} capacity {capacity}",
                    entry.query,
                    op.q_heads,
                    op.kv_heads,
                    op.v_dim,
                    cache_name(cache),
                    op.positions
                )
            };
            ops.push(rope_kv(name(CAPACITY), op, Need::rows(CAPACITY)));
            if computed(entry.query) {
                let name = format!("{} angles", name(CAPACITY));
                ops.push(rope_kv_angles(name, op, Need::rows(CAPACITY)));
            }
            if matches!(cache, KvWrite::Paged { .. }) && !wide.contains(&kv_dtype) {
                wide.push(kv_dtype);
                ops.push(rope_kv(name(WIDE_CAPACITY), op, Need::rows(WIDE_CAPACITY)));
            }
        } else if let Some(arguments) = template(&entry.body, "dflash_rope") {
            let heads = arguments[0]
                .parse()
                .map_err(|_| format!("{}: heads", entry.symbol))?;
            let op = RopeOp {
                rope: entry.query,
                heads,
                positions: positions(arguments[1]),
            };
            let name = format!(
                "rope {:?} heads {heads} {:?} capacity {CAPACITY}",
                entry.query, op.positions
            );
            ops.push(rope(name.clone(), op, Need::rows(CAPACITY)));
            if computed(entry.query) {
                ops.push(rope_angles(
                    format!("{name} angles"),
                    op,
                    Need::rows(CAPACITY),
                ));
            }
        } else if let Some(arguments) = template(&entry.body, "ring_write") {
            let op = RingWriteOp {
                query: entry.query,
                positions: positions(arguments[4]),
                single_sequence: arguments.get(5) == Some(&"true"),
            };
            let name = format!(
                "ring write {:?} {:?}{} capacity {CAPACITY}",
                entry.query,
                op.positions,
                if op.single_sequence {
                    " single sequence"
                } else {
                    ""
                }
            );
            ops.push(ring_write(name, op, Need::rows(CAPACITY)));
        }
    }
    Ok(ops)
}

/// Whether `query` is a rotation by computed angles.
fn computed(query: RowQuery) -> bool {
    matches!(
        query,
        RowQuery::Rope {
            source: RotarySource::Computed,
            ..
        }
    )
}

fn cache_name(cache: KvWrite) -> String {
    match cache {
        KvWrite::Paged {
            page_size,
            max_context,
            kv_dtype,
        } => format!("paged p{page_size} context {max_context} {kv_dtype:?}"),
        KvWrite::Ring {
            ring_size,
            kv_dtype,
            store,
        } => format!(
            "ring {ring_size} {kv_dtype:?}{}",
            if store { "" } else { " no store" }
        ),
    }
}

/// Self-checks of the family's cases over a consumer's RoPE and KV write
/// entries (as [`sm120_entries`] selects them), for its tests: each panics
/// naming the first failure. kernel-lab runs them over the in-tree
/// catalogs.
pub mod checks {
    use super::super::testing::{assert_maps_follow_rows, prepared, writes};
    use super::*;
    use std::collections::BTreeSet;

    /// The cache plane maps follow the rows the property checks move: row
    /// 8 moved to the front writes where row 8 wrote, not where row 0 did.
    pub fn plane_maps_follow_moved_rows(entries: &[RopeKvEntry]) {
        let backend = kernels_sm120::backend::Sm120::new().with_rope_kv(entries.to_vec());
        let ops = family(entries).unwrap();
        for form in ["paged p16", "ring write"] {
            let required = ops
                .iter()
                .find(|op| op.name.contains(form) && op.name.ends_with("capacity 16"))
                .unwrap_or_else(|| panic!("no {form} op"));
            let case = prepared(&backend, required);
            for plane in ["cache.k", "cache.v"] {
                assert_maps_follow_rows(&case, plane, 8);
            }
        }
    }

    /// The ring planes are `oracle::ring_kv_write`'s: at every live count
    /// they change only bytes the write maps name, and some at a positive
    /// count. Every element of both planes also equals a model of the ring
    /// layout and skip rule ([`ring_rows`]) written apart from `RingStore`:
    /// the ring write's BF16 bits or E4M3 codes of its K and V inputs, with
    /// its superseded rows unstored, and the fused write's K values and V
    /// bits or codes of its `k_new` and `v_new`.
    pub fn ring_planes_follow_the_ring_kv_write_oracle(entries: &[RopeKvEntry]) {
        let backend = kernels_sm120::backend::Sm120::new().with_rope_kv(entries.to_vec());
        let ops = family(entries).unwrap();
        let rings = ops
            .iter()
            .filter(|op| op.name.contains("ring") && !op.name.contains("no store"))
            .collect::<Vec<_>>();
        assert!(rings.iter().any(|op| op.name.starts_with("ring write")));
        assert!(rings.iter().any(|op| op.name.starts_with("rope kv")));
        for required in &rings {
            let case = prepared(&backend, required);
            let images = case.case.images();
            for live in [case.capacity, 5, 1, 0] {
                for (name, expected) in case.case.expected(&images, live).unwrap() {
                    let Expected::Bits(bits) = expected else {
                        continue;
                    };
                    if !name.starts_with("cache.") {
                        continue;
                    }
                    let image = images.bytes(&name).unwrap();
                    let changed = (0..bits.len())
                        .filter(|&at| bits[at] != image[at])
                        .collect::<BTreeSet<_>>();
                    let written = writes(&case, &name, &images, live);
                    assert!(
                        changed.is_subset(&written),
                        "{} {name} at live {live}",
                        required.name
                    );
                    assert_eq!(live > 0, !changed.is_empty(), "{} {name}", required.name);
                }
            }
        }

        // Every element of both planes, apart from `RingStore`.
        let ring = ring_size(entries).expect("a ring") as usize;
        let mut checked = BTreeSet::new();
        for required in &rings {
            let case = prepared(&backend, required);
            let images = case.case.images();
            let fp8 = required.name.contains("Fp8E4M3");
            let scales = if fp8 { FP8_SCALES } else { [1.0; 2] };
            let dtype = if fp8 { DType::Fp8E4M3 } else { DType::Bf16 };
            let positions = if required.name.contains(" I64") {
                DType::I64
            } else {
                DType::I32
            };
            let slots = images.i32s("slot_ids").unwrap();
            let all = case.capacity as usize;
            let fused = required.name.starts_with("rope kv");
            for live in [case.capacity, 5, 1, 0] {
                let rows = placed_rows(&images, positions).unwrap();
                let rows = &rows[..live as usize];
                let last = if fused {
                    let mut last = vec![0; slots.len()];
                    for &(item, position) in rows {
                        last[item] = last[item].max(position);
                    }
                    last
                } else {
                    positions_of(&images, "last_positions", DType::I64).unwrap()
                };
                let targets = ring_rows(rows, ring, &slots, &last);
                let superseded = targets.iter().filter(|t| t.is_none()).count();
                let expected = case
                    .case
                    .expected(&images, live)
                    .unwrap()
                    .into_iter()
                    .collect::<std::collections::HashMap<_, _>>();
                let bits = |name: &str| match &expected[name] {
                    Expected::Bits(bits) => bits.clone(),
                    _ => panic!("{} {name}: no bits", required.name),
                };
                let halves = |bytes: Vec<u8>| {
                    bytes
                        .chunks_exact(2)
                        .map(|half| u16::from_le_bytes([half[0], half[1]]))
                        .collect::<Vec<_>>()
                };
                let (k, v) = if fused {
                    (None, halves(bits("v_new")))
                } else {
                    (
                        Some(images.u16s("k_new").unwrap()),
                        images.u16s("v_new").unwrap(),
                    )
                };
                let v_plane = images.bytes("cache.v").unwrap();
                let want = ring_bits(v_plane, fp8, &targets, &v, v.len() / all, scales[1]);
                assert_eq!(bits("cache.v"), want, "{} V at live {live}", required.name);
                if let Some(k) = k {
                    let k_plane = images.bytes("cache.k").unwrap();
                    let want = ring_bits(k_plane, fp8, &targets, &k, k.len() / all, scales[0]);
                    assert_eq!(bits("cache.k"), want, "{} K at live {live}", required.name);
                    checked.insert(("K", fp8));
                } else if let Expected::Values(k) = &expected["k_new"] {
                    let Expected::Values(plane) = &expected["cache.k"] else {
                        panic!("{}: no K plane values", required.name);
                    };
                    let width = k.values.len() / all;
                    let (mut want, _) = plane_values(images.bytes("cache.k").unwrap(), dtype);
                    for (row, target) in targets.iter().enumerate() {
                        let at = target.expect("a fused row stored") * width;
                        for element in 0..width {
                            want[at + element] =
                                k.values[row * width + element] / f64::from(scales[0]);
                        }
                    }
                    assert_eq!(plane.values, want, "{} K at live {live}", required.name);
                    checked.insert(("fused K", fp8));
                }
                checked.insert((if fused { "fused V" } else { "V" }, fp8));
                // The two-item ring write supersedes its second item's first
                // three rows; the fused write stores every live row.
                if fused || live == case.capacity {
                    let two = !fused && !required.name.contains("single sequence");
                    let wanted = if two { 3 } else { 0 };
                    assert_eq!(superseded, wanted, "{} at live {live}", required.name);
                }
            }
        }
        for plane in ["K", "V", "fused K", "fused V"] {
            for fp8 in [false, true] {
                assert!(checked.contains(&(plane, fp8)), "{plane} FP8 {fp8}");
            }
        }
    }

    /// Each row's ring row, `slots[item] * ring + position % ring`, or none
    /// for a row a ring or more before its item's `last` position: the ring
    /// write's layout and skip rule, apart from [`RingStore`].
    fn ring_rows(
        rows: &[(usize, usize)],
        ring: usize,
        slots: &[i32],
        last: &[usize],
    ) -> Vec<Option<usize>> {
        rows.iter()
            .map(|&(item, position)| {
                (position + ring > last[item])
                    .then(|| slots[item] as usize * ring + position % ring)
            })
            .collect()
    }

    /// The plane `image` with each stored row of `width` BF16 `values` at
    /// its ring row: the bits, or the E4M3 codes of `value / scale`.
    fn ring_bits(
        image: &[u8],
        fp8: bool,
        targets: &[Option<usize>],
        values: &[u16],
        width: usize,
        scale: f32,
    ) -> Vec<u8> {
        let mut plane = image.to_vec();
        for (row, target) in targets.iter().enumerate() {
            let Some(target) = target else {
                continue;
            };
            for (element, &bits) in values[row * width..][..width].iter().enumerate() {
                let at = target * width + element;
                if fp8 {
                    plane[at] = SmallFloat::E4M3.encode_f32(bf16_to_f32(bits) / scale);
                } else {
                    plane[at * 2..][..2].copy_from_slice(&bits.to_le_bytes());
                }
            }
        }
        plane
    }

    /// Every computed rotation of the family has an angle op, whose case
    /// judges each rotated output, and no table rotation has one.
    pub fn every_computed_rotation_has_an_angle_case(entries: &[RopeKvEntry]) {
        let backend = kernels_sm120::backend::Sm120::new().with_rope_kv(entries.to_vec());
        let ops = family(entries).unwrap();
        let angles = ops
            .iter()
            .filter(|op| op.name.ends_with(" angles"))
            .collect::<Vec<_>>();
        let computed = ops
            .iter()
            .filter(|op| {
                op.name.contains("Computed") && op.name.ends_with(&format!("capacity {CAPACITY}"))
            })
            .count();
        assert!(computed > 0);
        assert_eq!(angles.len(), computed);
        assert!(angles.iter().any(|op| op.name.starts_with("rope Rope")));
        for required in angles {
            assert!(required.name.contains("Computed"), "{}", required.name);
            let case = prepared(&backend, required);
            let judged = case
                .case
                .expected(&case.case.images(), case.capacity)
                .unwrap()
                .into_iter()
                .filter(|(_, expected)| matches!(expected, Expected::Judge(_)))
                .map(|(name, _)| name)
                .collect::<Vec<_>>();
            assert!(
                judged == ["q", "k_new"] || judged == ["output"],
                "{}: {judged:?}",
                required.name
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::rotary::{AngleRule, Axes, ComputedTable};

    /// The angle judge admits unit pairs rotated by BF16 cosines and sines
    /// of the exact angles or of the models' HF table, and refuses a skipped
    /// rotation, the next pair's or the next position's angles, and a
    /// changed dimension past the rotation.
    #[test]
    fn the_angle_judge_admits_bf16_cosines_and_sines_only() {
        for (style, rotary, theta) in [
            (RotaryStyle::Neox, 64, 1e4),
            (RotaryStyle::Neox, 64, 1e7),
            (RotaryStyle::GptJ, 128, 1e4),
        ] {
            let rope = Rope {
                head_dim: 192,
                rotary,
                theta,
                style,
                table: false,
            };
            let (heads, pairs) = (2, rotary / 2);
            let width = heads * rope.head_dim;
            let mut input = bf16_values(5, ANGLE_POSITIONS.len() * width);
            rope.unit_pairs(&mut input, width, heads);
            let device = |angle: &dyn Fn(usize, usize) -> (f64, f64)| {
                let mut output = input.clone();
                for (row, &position) in ANGLE_POSITIONS.iter().enumerate() {
                    for head in 0..heads {
                        let at = row * width + head * rope.head_dim;
                        for dim in 0..rotary {
                            if let Some(mate) = rope.mate(dim) {
                                let (c, s) = angle(rope.pair(dim), position);
                                output[at + dim] = f32_to_bf16(c as f32);
                                output[at + mate] = f32_to_bf16(s as f32);
                            }
                        }
                    }
                }
                bf16_bytes(&output)
            };
            let exact = |pair: usize, position: usize| {
                let angle = position as f64 / theta.powf(2.0 * pair as f64 / rotary as f64);
                (angle.cos(), angle.sin())
            };
            let judge = rope.angle_judge(input.clone(), heads, ANGLE_POSITIONS.to_vec());
            judge(&device(&exact)).unwrap();
            if rotary == 64 {
                let table = ComputedTable {
                    theta,
                    rule: AngleRule::HfF32AngleBf16CosSin,
                    axes: Axes::Linear,
                };
                judge(&device(&|pair, position| {
                    table.cos_sin([position; 3], pair, pairs).unwrap()
                }))
                .unwrap();
            }
            assert!(judge(&device(&|_, _| (1.0, 0.0))).is_err());
            assert!(
                judge(&device(&|pair, position| exact(
                    (pair + 1) % pairs,
                    position
                )))
                .is_err()
            );
            assert!(judge(&device(&|pair, position| exact(pair, position + 1))).is_err());
            let mut changed = device(&exact);
            changed[2 * rotary] ^= 1;
            assert!(judge(&changed).is_err());
            assert!(judge(&changed[..changed.len() - 2]).is_err());
        }
    }

    #[test]
    fn rows_alternate_items_at_consecutive_positions() {
        assert_eq!(
            placement(5, &[13, 33]),
            vec![(0, 13), (1, 33), (0, 14), (1, 34), (0, 15)]
        );
        assert_eq!(placement(1, &[13, 33]), vec![(0, 13)]);
        // The wide paged case's last position fits its context.
        let last = 2 * 16 + 1 + WIDE_CAPACITY.div_ceil(2) - 1;
        assert!(last < paged_context(16, WIDE_CAPACITY));
    }

    /// The bound allows a BF16 rounding of each cosine and sine, and no
    /// more: a rotation through BF16 cosines and sines passes, one that
    /// misses a pair's sine fails.
    #[test]
    fn the_rotation_bound_admits_bf16_cosines_and_sines_only() {
        let rope = Rope {
            head_dim: 8,
            rotary: 4,
            theta: 1e4,
            style: RotaryStyle::Neox,
            table: false,
        };
        let x = widen(&bf16_values(3, 8));
        let positions = [300];
        let (values, allowance) = rope
            .apply(&Images::default(), "cos_sin", &x, 1, &positions)
            .unwrap();
        let rounded = |v: f64| f64::from(bf16_to_f32(kernel_core::codec::f32_to_bf16(v as f32)));
        let device = |drop_sine: bool| {
            (0..8)
                .map(|at| {
                    if at >= 4 {
                        return x[at];
                    }
                    let pair = at % 2;
                    let angle = 300.0 / 1e4_f64.powf(2.0 * pair as f64 / 4.0);
                    let (c, s) = (rounded(angle.cos()), rounded(angle.sin()));
                    let s = if drop_sine { 0.0 } else { s };
                    if at < 2 {
                        x[at] * c - x[at + 2] * s
                    } else {
                        x[at] * c + x[at - 2] * s
                    }
                })
                .map(|v| rounded(v) as f32)
                .collect::<Vec<_>>()
        };
        let Expected::Values(reference) = bounded(values, &allowance).unwrap() else {
            unreachable!("values");
        };
        let check = |actual: &[f32]| {
            crate::conformance::rounding_admissible(DType::Bf16, actual, &reference)
        };
        check(&device(false)).unwrap();
        assert!(check(&device(true)).is_err());
    }

    /// A value just below a binade boundary whose allowance crosses it may
    /// round by a step of the next binade (GPU: a computed RoPE q of
    /// 0.99968 within 0.0048 read back one step above 1.0039).
    #[test]
    fn rounding_past_a_binade_boundary_is_admitted() {
        let (reference, allowance) = (0.999_684_7, 0.004_811);
        // A device value inside the allowance, past 1 + half a step there,
        // rounds to 1 + one step of [1, 2).
        let device = f64::from(bf16_to_f32(kernel_core::codec::f32_to_bf16(
            (reference + 0.004_7) as f32,
        )));
        assert_eq!(device, 1.0 + 2_f64.powi(-7));
        check_bf16(vec![reference], &[allowance], &[device as f32]).unwrap();
        // Two steps past the allowance still fail.
        assert!(
            check_bf16(
                vec![reference],
                &[allowance],
                &[(1.0 + 2_f64.powi(-6)) as f32]
            )
            .is_err()
        );
    }

    /// The gate admits only the BF16 values that round the allowance's
    /// ends: both ends of 0.9999 within 0.0002 round to 1, so the value
    /// one step below fails, though it is within the allowance and half a
    /// step at 0.9999.
    #[test]
    fn only_the_roundings_of_the_allowance_are_admitted() {
        check_bf16(vec![0.9999], &[0.0002], &[1.0]).unwrap();
        assert!(check_bf16(vec![0.9999], &[0.0002], &[1.0 - 2_f32.powi(-8)]).is_err());
    }

    fn table_rope(head_dim: usize, rotary: usize) -> Rope {
        Rope {
            head_dim,
            rotary,
            theta: 1e4,
            style: RotaryStyle::Neox,
            table: true,
        }
    }

    fn check_bf16(values: Vec<f64>, allowance: &[f64], actual: &[f32]) -> LabResult<()> {
        let Expected::Values(reference) = bounded(values, allowance)? else {
            unreachable!("values");
        };
        crate::conformance::rounding_admissible(DType::Bf16, actual, &reference)
    }

    /// A supplied table fixes the coefficients: the identity rotation of
    /// [1, 1] gives [1, 1], and one BF16 step off fails.
    #[test]
    fn a_table_rotation_is_bound_by_its_arithmetic_alone() {
        let rope = table_rope(2, 2);
        let images = Images::default().with(
            "cos_sin",
            [1.0_f32, 0.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        );
        let (values, allowance) = rope
            .apply(&images, "cos_sin", &[1.0, 1.0], 1, &[0])
            .unwrap();
        check_bf16(values.clone(), &allowance, &[1.0, 1.0]).unwrap();
        assert!(check_bf16(values, &allowance, &[1.007_812_5, 1.0]).is_err());
    }

    /// The case's tables are not the computed angles: a launch that ignores
    /// the table and computes them fails every row.
    #[test]
    fn tables_differ_from_the_computed_angles() {
        let rope = table_rope(128, 64);
        let positions = (5..21).collect::<Vec<_>>();
        let images = Images::default().with("cos_sin", rope.table_image(21));
        let x = widen(&bf16_values(3, positions.len() * 128));
        let (values, allowance) = rope.apply(&images, "cos_sin", &x, 1, &positions).unwrap();
        let computed = Rope {
            table: false,
            ..rope
        };
        let (ignored, _) = computed
            .apply(&images, "cos_sin", &x, 1, &positions)
            .unwrap();
        for (row, (values, ignored)) in values.chunks(128).zip(ignored.chunks(128)).enumerate() {
            let ignored = ignored.iter().map(|&v| v as f32).collect::<Vec<_>>();
            assert!(
                check_bf16(values.to_vec(), &allowance[row * 128..][..128], &ignored).is_err(),
                "row {row}"
            );
        }
    }
}
