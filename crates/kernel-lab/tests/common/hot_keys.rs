//! One-hot attention operands for conformance cases whose outputs must be
//! exact bits, so a high-id rerun can show that a launch reads the pages
//! its IDs name.
//!
//! Every token of a pool holds a distinct code below [`CODES`]. Its key row
//! is BF16 `[c / 64 + 1, c % 64 + 1, filler...]` and a query head's row
//! `[+-2^18, +-2^12, 0...]`, so their dot is `+-(2^18 (c / 64 + 1) + 2^12
//! (c % 64 + 1))`: every product and partial sum is a multiple of 2^12
//! below 2^25, which F32 holds exactly in any order, and two codes' dots
//! differ by at least 4096. At a softmax scale of 1/16 the scores differ by
//! at least 256, so every key but the best has `exp(s - m)` below the
//! smallest F32 subnormal: the softmax is exactly one-hot, the denominator
//! is 1 and the numerator is the best key's value row. A positive head picks
//! the largest code it sees and a negative head the smallest. The oracle
//! checks this on the operands it is given and refuses a key set outside
//! it: a non-finite element, a dot F32 cannot sum exactly and finitely in
//! every order, F32 scores within [`MARGIN`] of the best, or scores so far
//! apart that an F32 exponent argument overflows
//! (`tests/hot_keys_refusal.rs`).
#![allow(dead_code)] // Shared by test binaries that use different parts.

use std::f32::consts::LOG2_E;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use kernel_core::codec::bf16_to_f32;
use kernel_lab::LabResult;
use kernel_lab::conformance::{Expected, Images};

/// Codes of a pool's tokens: every code lies below this, and a pool of
/// more tokens repeats them page by page.
pub const CODES: usize = 4096;
/// Pages of a pool whose high-id rerun moves an entry's pages to IDs past 16
/// bits. An entry whose `u` pages hold IDs `0..u`, as these cases' do,
/// moves page `j` to `65636 - u + j`, and the rerun shifts every other page
/// down by `u`. A launch that narrows that ID to 16 bits reads the page
/// that was `100 + j`, and one that ignores the moved IDs the page that was
/// `u + j`; for `u` below 64, neither 100 nor `u` is a multiple of the pages
/// after which codes repeat (64 here, 16 for pooled keys), so neither holds
/// page `j`'s codes ([`narrowing_checked`] shows this per case). Byte
/// offsets pass 2^31 too wherever a page takes 32 KiB or more.
pub const WIDE_PAGES: usize = (1 << 16) + 100;
/// The smallest F32 score gap [`hot`] accepts: `exp(-128)` lies far below
/// the smallest F32 subnormal, so the runner-up's probability is +0.
pub const MARGIN: f64 = 128.0;
/// BF16 decoy between pages, 3.4e38, which turns any read outside a page
/// into a score the oracle does not predict.
pub const DECOY: u8 = 0x7f;

/// A query head's row of `width` BF16: dims 0 and 1 hold 2^18 and 2^12,
/// negated when `negative`, and every other dim +0.
pub fn query_row(width: usize, negative: bool) -> Vec<u16> {
    let sign = if negative { 0x8000 } else { 0 };
    let mut row = vec![0_u16; width];
    // 2^18 and 2^12 as BF16.
    row[0] = 0x4880 | sign;
    row[1] = 0x4580 | sign;
    row
}

/// A small positive integer as BF16 (exact up to 256).
fn small(value: usize) -> u16 {
    assert!((1..=256).contains(&value), "{value} is not a small integer");
    ((value as f32).to_bits() >> 16) as u16
}

/// Token `code`'s key row of `width` BF16: `[code / 64 + 1, code % 64 + 1]`
/// then positive filler, every element a nonzero integer.
pub fn key_row(code: usize, width: usize) -> Vec<u16> {
    assert!(code < CODES, "code {code} is not below {CODES}");
    (0..width)
        .map(|d| match d {
            0 => small(code / 64 + 1),
            1 => small(code % 64 + 1),
            _ => small((code * 7 + d) % 61 + 1),
        })
        .collect()
}

/// Token `code`'s value row of `width` BF16: nonzero integers, distinct
/// from its key row and from every other code's value row in dims 0 and 1.
pub fn value_row(code: usize, width: usize) -> Vec<u16> {
    assert!(code < CODES, "code {code} is not below {CODES}");
    (0..width)
        .map(|d| match d {
            0 => small(code / 64 + 3),
            1 => small(code % 64 + 5),
            _ => small((code * 3 + d * 5) % 59 + 1),
        })
        .collect()
}

pub fn bf16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The values of a BF16 image.
pub fn bf16_values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|half| f64::from(bf16_to_f32(u16::from_le_bytes([half[0], half[1]]))))
        .collect()
}

/// A paged pool of `pages` pages at `stride` bytes, each `page` tokens of
/// `width` BF16 from `row` of the token's code `p % (CODES / page) * page +
/// token` for page `p` (distinct codes in any `CODES / page` consecutive
/// pages), then decoy bytes to the stride.
pub fn pool(
    pages: usize,
    page: usize,
    width: usize,
    stride: usize,
    row: fn(usize, usize) -> Vec<u16>,
) -> Vec<u8> {
    assert!(
        CODES.is_multiple_of(page),
        "{page}-token pages do not divide {CODES} codes"
    );
    assert!(page * width * 2 <= stride, "a page overflows its stride");
    // The pages of distinct codes, repeated.
    let distinct = pages.min(CODES / page);
    let mut pattern = vec![DECOY; distinct * stride];
    for p in 0..distinct {
        for t in 0..page {
            let bytes = bf16_bytes(&row(p * page + t, width));
            pattern[p * stride + t * width * 2..][..bytes.len()].copy_from_slice(&bytes);
        }
    }
    let mut image = Vec::with_capacity(pages * stride);
    while image.len() < pages * stride {
        let more = (pages * stride - image.len()).min(pattern.len());
        image.extend_from_slice(&pattern[..more]);
    }
    image
}

/// An ID as a launch that keeps its low 16 bits reads it; holes stay.
pub fn low16(id: i32) -> i32 {
    if id < 0 { id } else { id & 0xffff }
}

/// An ID as a launch that keeps it whole reads it.
pub fn whole(id: i32) -> i32 {
    id
}

/// Evaluations of [`narrowing_checked`] whose narrowed IDs changed an exact
/// output, and the largest ID among them.
pub static NARROWED: AtomicUsize = AtomicUsize::new(0);
pub static HIGHEST: AtomicI32 = AtomicI32::new(-1);

/// The oracle `outputs(images, live, whole)`, where the last argument maps
/// each ID of `ids` to the page the launch reads. When `ids` holds an ID
/// past 16 bits, as in the high-id rerun over [`WIDE_PAGES`], it also
/// evaluates `outputs(images, live, low16)`, the read of a launch that
/// narrows the IDs, and refuses unless an exact output changes, so the
/// rerun's bitwise comparison refuses such a launch.
pub fn narrowing_checked(
    images: &Images,
    live: u32,
    ids: &str,
    outputs: impl Fn(&Images, u32, fn(i32) -> i32) -> LabResult<Vec<(String, Expected)>>,
) -> LabResult<Vec<(String, Expected)>> {
    let exact = outputs(images, live, whole)?;
    let Some(highest) = images
        .i32s(ids)?
        .into_iter()
        .filter(|&id| id > 0xffff)
        .max()
    else {
        return Ok(exact);
    };
    let narrowed =
        outputs(images, live, low16).map_err(|e| format!("{ids} narrowed to 16 bits: {e}"))?;
    let changed = exact.iter().zip(&narrowed).any(|(exact, narrowed)| {
        matches!((&exact.1, &narrowed.1), (Expected::Bits(a), Expected::Bits(b)) if a != b)
    });
    if !changed {
        return Err(format!(
            "{ids} narrowed to 16 bits leaves every exact output at live {live} unchanged, so a \
             launch that narrows them passes the high-id rerun"
        ));
    }
    NARROWED.fetch_add(1, Ordering::Relaxed);
    HIGHEST.fetch_max(highest, Ordering::Relaxed);
    Ok(exact)
}

/// The BF16 row of `width` at token `slot` of page `id` of `pool`.
pub fn token(pool: &[u8], stride: usize, width: usize, id: i32, slot: usize) -> LabResult<&[u8]> {
    let id = usize::try_from(id).map_err(|_| format!("page {id} is a hole"))?;
    pool.get(id * stride + slot * width * 2..)
        .and_then(|bytes| bytes.get(..width * 2))
        .ok_or_else(|| format!("page {id} slot {slot} lies past the pool"))
}

/// The dot of a query row with a key row over every dim, as a kernel sums
/// it in F32: refused unless every element is finite (a kernel multiplies
/// the dims a query zeroes too) and every product and partial sum is exact
/// and finite in F32 in any order. Each is a multiple of the lowest set bit
/// of any product and at most the products' magnitudes summed, so this
/// holds when that bit is no smaller than the smallest F32 subnormal (the
/// kernels keep subnormals), the magnitudes sum below 2^24 times it, and
/// that sum is at most the largest F32. `query` holds decoded BF16, so each
/// product is exact in F64.
pub fn dot(query: &[f64], key: &[u8]) -> LabResult<f64> {
    dot_values(query, &bf16_values(key))
}

/// [`dot`] over a key row of decoded values, which each product must hold
/// exactly in F64 (true of any two F32 values).
pub fn dot_values(query: &[f64], key: &[f64]) -> LabResult<f64> {
    if query.len() != key.len() {
        return Err(format!(
            "a query row of {} dims against a key row of {}",
            query.len(),
            key.len()
        ));
    }
    let (mut sum, mut magnitude, mut unit) = (0.0, 0.0, f64::INFINITY);
    for (d, (q, k)) in query.iter().zip(key).enumerate() {
        if !q.is_finite() || !k.is_finite() {
            return Err(format!(
                "dim {d} multiplies {q} by {k}, which is not finite, so the dot has no exact bits"
            ));
        }
        let product = q * k;
        if product != 0.0 {
            sum += product;
            magnitude += product.abs();
            unit = f64::min(unit, lowest_bit(product));
        }
    }
    if magnitude > f64::from(f32::MAX) {
        return Err(format!(
            "a dot of products {magnitude} in magnitude overflows F32 in some order"
        ));
    }
    if unit < 2_f64.powi(-149) {
        return Err(format!(
            "a dot of products in steps of {unit}, below the smallest F32 subnormal, is not \
             exactly an F32 sum"
        ));
    }
    if magnitude >= unit * 2_f64.powi(24) {
        return Err(format!(
            "a dot of products {magnitude} in magnitude, in steps of {unit}, is not exactly \
             an F32 sum in every order"
        ));
    }
    Ok(sum)
}

/// The value of the lowest set bit of a finite nonzero F64.
fn lowest_bit(value: f64) -> f64 {
    let bits = value.abs().to_bits();
    let exponent = (bits >> 52) as i32;
    let fraction = bits & ((1 << 52) - 1);
    let (significand, exponent) = if exponent == 0 {
        (fraction, 1)
    } else {
        (fraction | 1 << 52, exponent)
    };
    2_f64.powi(exponent - 1075 + significand.trailing_zeros() as i32)
}

/// The position of the one key whose F32 score `dot * scale` ([`score`],
/// which a kernel forms from an exact dot) beats every other by at least
/// [`MARGIN`], or None for no key. Refused when a score is not finite or
/// two come closer, since the softmax would then not be one-hot; the gap is
/// that of the rounded scores, which a kernel subtracts. Also refused when
/// the lowest score's F32 exponent argument `(score - top) * log2(e)`
/// overflows: every running maximum a kernel subtracts lies between the
/// lowest and the best score, so no other argument overflows.
pub fn hot(dots: &[f64], scale: f64) -> LabResult<Option<usize>> {
    let scores = dots
        .iter()
        .map(|&dot| score(dot, scale as f32))
        .collect::<Vec<_>>();
    if let Some(at) = scores.iter().position(|score| !score.is_finite()) {
        return Err(format!(
            "key {at}'s F32 score {} is not finite, so the softmax is not one-hot",
            scores[at]
        ));
    }
    let Some((best, &top)) = scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)) else {
        return Ok(None);
    };
    if let Some(far) = scores
        .iter()
        .position(|&score| !((score - top) * LOG2_E).is_finite())
    {
        return Err(format!(
            "key {far}'s F32 score {} lies so far below {top} that its exponent argument \
             overflows F32",
            scores[far]
        ));
    }
    if let Some(close) = scores
        .iter()
        .enumerate()
        .find(|&(at, &score)| at != best && f64::from(top) - f64::from(score) < MARGIN)
    {
        return Err(format!(
            "keys {best} and {} score within {MARGIN} in F32, so the softmax is not one-hot",
            close.0
        ));
    }
    Ok(Some(best))
}

/// The F32 score `dot * scale` the kernels form with one rounding; exact
/// for these operands and power-of-two scales.
pub fn score(dot: f64, scale: f32) -> f32 {
    (dot as f32) * scale
}

/// One row's F32 attention partials over its kept keys: the maximum and
/// denominator `[heads]` and the numerator `[heads, width]`.
pub struct Partials {
    pub max: Vec<f32>,
    pub den: Vec<f32>,
    pub num: Vec<f32>,
}

/// The partials of `heads` query rows of `width` (`queries`, decoded) over
/// the kept keys `(page id, slot)` of the K and V pools at `stride`, with
/// unit K and V scales: the best key's score, 1 and its value row, exact
/// since the softmax is one-hot. Without a kept key, -inf, +0 and +0.
#[allow(clippy::too_many_arguments)]
pub fn partials(
    queries: &[f64],
    heads: usize,
    width: usize,
    (k, v): (&[u8], &[u8]),
    stride: usize,
    keys: &[(i32, usize)],
    scale: f32,
) -> LabResult<Partials> {
    let mut partials = Partials {
        max: vec![f32::NEG_INFINITY; heads],
        den: vec![0.0; heads],
        num: vec![0.0; heads * width],
    };
    let rows = keys
        .iter()
        .map(|&(id, slot)| token(k, stride, width, id, slot))
        .collect::<LabResult<Vec<_>>>()?;
    // A kernel weighs every kept value row, the losers by +0, which makes
    // NaN of an infinite element.
    for &(id, slot) in keys {
        if bf16_values(token(v, stride, width, id, slot)?)
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(format!(
                "page {id} slot {slot}'s value row is not finite, so its +0 weight makes NaN"
            ));
        }
    }
    for head in 0..heads {
        let query = &queries[head * width..][..width];
        let dots = rows
            .iter()
            .map(|row| dot(query, row))
            .collect::<LabResult<Vec<_>>>()?;
        let Some(best) = hot(&dots, f64::from(scale))? else {
            continue;
        };
        let (id, slot) = keys[best];
        partials.max[head] = score(dots[best], scale);
        partials.den[head] = 1.0;
        for (out, value) in partials.num[head * width..][..width]
            .iter_mut()
            .zip(bf16_values(token(v, stride, width, id, slot)?))
        {
            *out = value as f32;
        }
    }
    Ok(partials)
}

/// The paged and selected partial entries' shared layout: query heads, head
/// width, page tokens, page stride and softmax scale.
#[derive(Clone, Copy)]
pub struct PagedShape {
    pub heads: usize,
    pub width: usize,
    pub page: usize,
    pub stride: usize,
    pub scale: f32,
}

/// The exact partials `out_max`, `out_den` and `out_num` of a paged or
/// selected partial entry at `live` of `capacity` rows, from its images
/// (`q`, `k_cache`, `v_cache`, `block_tables` of `max_pages` per item,
/// `row_batch`, `positions`) at group size 1, reading each table ID as
/// `read` maps it. `kept(images, row, position)` lists the positions row
/// `row` attends.
#[allow(clippy::too_many_arguments)]
pub fn partial_outputs(
    images: &Images,
    live: u32,
    capacity: usize,
    shape: PagedShape,
    max_pages: usize,
    read: fn(i32) -> i32,
    kept: impl Fn(&Images, usize, usize) -> LabResult<Vec<usize>>,
) -> LabResult<Vec<(String, Expected)>> {
    let PagedShape {
        heads,
        width,
        page,
        stride,
        scale,
    } = shape;
    let q = bf16_values(images.bytes("q")?);
    let pools = (images.bytes("k_cache")?, images.bytes("v_cache")?);
    let tables = images.i32s("block_tables")?;
    let (items, positions) = (images.i32s("row_batch")?, images.i32s("positions")?);
    let mut max = vec![0.0; capacity * heads];
    let mut den = vec![0.0; capacity * heads];
    let mut num = vec![0.0; capacity * heads * width];
    for row in 0..live as usize {
        let item = usize::try_from(items[row]).map_err(|e| e.to_string())?;
        let position = usize::try_from(positions[row]).map_err(|e| e.to_string())?;
        let keys = kept(images, row, position)?
            .into_iter()
            .map(|p| {
                let id = tables
                    .get(item * max_pages + p / page)
                    .ok_or_else(|| format!("row {row} position {p} lies past its table"))?;
                Ok((read(*id), p % page))
            })
            .collect::<LabResult<Vec<_>>>()?;
        let partial = partials(
            &q[row * heads * width..][..heads * width],
            heads,
            width,
            pools,
            stride,
            &keys,
            scale,
        )?;
        max[row * heads..][..heads].copy_from_slice(&partial.max);
        den[row * heads..][..heads].copy_from_slice(&partial.den);
        num[row * heads * width..][..heads * width].copy_from_slice(&partial.num);
    }
    Ok(vec![
        ("out_max".into(), Expected::Bits(f32_bytes(&max))),
        ("out_den".into(), Expected::Bits(f32_bytes(&den))),
        ("out_num".into(), Expected::Bits(f32_bytes(&num))),
    ])
}
