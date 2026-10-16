//! FP8 E4M3 one-hot attention operands: `common/hot_keys.rs` for entries
//! over FP8 KV, whose conformance cases must give exact bits too.
//!
//! E4M3 holds integers exactly only up to 16, so `hot_keys.rs`' two digits
//! of 64 do not fit. A key row here spells its code `c` below [`CODES`] in
//! three digits of 16, `[c / 256 + 1, c / 16 % 16 + 1, c % 16 + 1,
//! filler...]`, every element an integer in 1..=16, and a query head's BF16
//! row is `[+-2^20, +-2^16, +-2^12, 0...]`. Their dot is `+-(2^20 (c / 256
//! + 1) + 2^16 (c / 16 % 16 + 1) + 2^12 (c % 16 + 1))`: every product and
//! partial sum is a multiple of 2^12 below 2^25, which F32 holds exactly in
//! any order, and a digit's step outweighs every lower digit's 15 steps by
//! 2^12 (`2^20 - 15 (2^16 + 2^12) = 2^12`), so two codes' dots differ by at
//! least 4096. The kernels scale the dot by the K scale: the paged and
//! selected partials by `k_scale * scale` ([`K_SCALE`] 2 at a softmax
//! scale of 1/16 leaves scores at least 512 apart), latent attention each
//! 128-wide block's chain by that token's block scale. The softmax is then
//! one-hot as in `hot_keys.rs` ([`hot`], which refuses scores within
//! `MARGIN`), and the numerator is the best key's value row times the V
//! scale: [`V_SCALE`] for the partials, and for latent attention, whose
//! values are its keys, each block's scale. Latent block 0 (the dims a
//! query reads) takes [`K_SCALE`] and every later block a power of two
//! that varies by token, so the outputs show which token's scales a launch
//! reads. The oracle refuses operands outside this: a non-finite or
//! inexact dot ([`dot_values`]), a latent block scale that is not a power
//! of two, or a value the F32 numerator does not hold exactly
//! (`tests/hot_fp8_refusal.rs`).
//!
//! A pool fills the bytes past each page with [`DECOY`], E4M3 NaN, so a
//! read outside a page makes the scores NaN, and through `exp` every
//! denominator.
#![allow(dead_code)] // Shared by test binaries that use different parts.

use crate::hot_keys::{CODES, DECOY, PagedShape, Partials, dot_values, f32_bytes, hot, score};
use kernel_core::codec::SmallFloat;
use kernel_lab::LabResult;
use kernel_lab::conformance::{Expected, Images};

/// The K scale of the partial entries (`k_scale_bits`) and of latent block
/// 0.
pub const K_SCALE: f32 = 2.0;
/// The V scale of the partial entries (`v_scale_bits`).
pub const V_SCALE: f32 = 0.25;
/// Latent dims per block scale (`LATENT_FP8_SCALE_BLOCK`).
pub const BLOCK: usize = 128;

/// `value` as E4M3, which it must hold exactly.
pub fn e4m3(value: f64) -> LabResult<u8> {
    let code = SmallFloat::E4M3.encode_f32(value as f32);
    if SmallFloat::E4M3.decode_f64(code) != value {
        return Err(format!("{value} is not an E4M3 value"));
    }
    Ok(code)
}

fn exact(value: f64) -> u8 {
    e4m3(value).unwrap_or_else(|e| panic!("{e}"))
}

/// The values of an E4M3 image.
pub fn fp8_values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .iter()
        .map(|&code| SmallFloat::E4M3.decode_f64(code))
        .collect()
}

/// A query head's row of `width` BF16: dims 0, 1 and 2 hold 2^20, 2^16 and
/// 2^12, negated when `negative`, and every other dim +0.
pub fn query_row(width: usize, negative: bool) -> Vec<u16> {
    let sign = if negative { 0x8000 } else { 0 };
    let mut row = vec![0_u16; width];
    // 2^20, 2^16 and 2^12 as BF16.
    row[0] = 0x4980 | sign;
    row[1] = 0x4780 | sign;
    row[2] = 0x4580 | sign;
    row
}

/// The digits of `code`, most significant first.
fn digits(code: usize) -> [usize; 3] {
    assert!(code < CODES, "code {code} is not below {CODES}");
    [code / 256, code / 16 % 16, code % 16]
}

/// Token `code`'s key row of `width` E4M3: its digits plus 1, then positive
/// integer filler, every element in 1..=16.
pub fn key_row(code: usize, width: usize) -> Vec<u8> {
    let digits = digits(code);
    (0..width)
        .map(|d| match d {
            0..3 => exact((digits[d] + 1) as f64),
            _ => exact(((code * 7 + d) % 15 + 1) as f64),
        })
        .collect()
}

/// Token `code`'s value row of `width` E4M3: 16 minus its digits, distinct
/// from its key row and from every other code's value row in dims 0 to 2,
/// then eighths of either sign.
pub fn value_row(code: usize, width: usize) -> Vec<u8> {
    let digits = digits(code);
    (0..width)
        .map(|d| match d {
            0..3 => exact((16 - digits[d]) as f64),
            _ => {
                let eighths = ((code * 3 + d * 5) % 15 + 1) as f64 / 8.0;
                exact(if (code + d) % 2 == 1 {
                    -eighths
                } else {
                    eighths
                })
            }
        })
        .collect()
}

/// Token `code`'s latent block scales over `width` dims: [`K_SCALE`] for
/// block 0, then 2^-3 to 2^3 by code and block.
pub fn latent_scales(code: usize, width: usize) -> Vec<f32> {
    (0..width / BLOCK)
        .map(|b| match b {
            0 => K_SCALE,
            _ => 2_f32.powi(((code * 5 + b) % 7) as i32 - 3),
        })
        .collect()
}

/// A paged pool of `pages` pages at `stride` bytes: page `p` holds
/// `page_image(first)` for its tokens' codes `first..first + page`, with
/// `first = p % (CODES / page) * page` (distinct codes in any `CODES /
/// page` consecutive pages), then decoy bytes to the stride.
pub fn pages_of(
    pages: usize,
    page: usize,
    stride: usize,
    page_image: impl Fn(usize) -> Vec<u8>,
) -> Vec<u8> {
    assert!(
        CODES.is_multiple_of(page),
        "{page}-token pages do not divide {CODES} codes"
    );
    // The pages of distinct codes, repeated.
    let distinct = pages.min(CODES / page);
    let mut pattern = vec![DECOY; distinct * stride];
    for p in 0..distinct {
        let bytes = page_image(p * page);
        assert!(bytes.len() <= stride, "a page overflows its stride");
        pattern[p * stride..][..bytes.len()].copy_from_slice(&bytes);
    }
    let mut image = Vec::with_capacity(pages * stride);
    while image.len() < pages * stride {
        let more = (pages * stride - image.len()).min(pattern.len());
        image.extend_from_slice(&pattern[..more]);
    }
    image
}

/// A paged K or V pool of rows `[page][width]` E4M3 from `row`.
pub fn pool(
    pages: usize,
    page: usize,
    width: usize,
    stride: usize,
    row: fn(usize, usize) -> Vec<u8>,
) -> Vec<u8> {
    pages_of(pages, page, stride, |first| {
        (first..first + page)
            .flat_map(|code| row(code, width))
            .collect()
    })
}

/// A latent pool in `LatentKvFormat::Fp8E4M3Block128` pages: every token's
/// codes `[page][width]` ([`key_row`]), then its block scales `[page][width
/// / 128]` F32 ([`latent_scales`]).
pub fn latent_pool(pages: usize, page: usize, width: usize, stride: usize) -> Vec<u8> {
    pages_of(pages, page, stride, |first| {
        let codes = first..first + page;
        let mut image = codes
            .clone()
            .flat_map(|code| key_row(code, width))
            .collect::<Vec<_>>();
        for code in codes {
            image.extend(f32_bytes(&latent_scales(code, width)));
        }
        image
    })
}

/// The E4M3 row of `width` at token `slot` of page `id` of `pool`.
pub fn token(pool: &[u8], stride: usize, width: usize, id: i32, slot: usize) -> LabResult<&[u8]> {
    let id = usize::try_from(id).map_err(|_| format!("page {id} is a hole"))?;
    pool.get(id * stride + slot * width..)
        .and_then(|bytes| bytes.get(..width))
        .ok_or_else(|| format!("page {id} slot {slot} lies past the pool"))
}

/// The decoded codes and block scales of token `slot` of page `id` of a
/// latent pool of `page`-token pages.
pub fn latent_token(
    pool: &[u8],
    stride: usize,
    page: usize,
    width: usize,
    id: i32,
    slot: usize,
) -> LabResult<(Vec<f64>, Vec<f32>)> {
    let id = usize::try_from(id).map_err(|_| format!("page {id} is a hole"))?;
    latent_token_at(pool, id * stride, page, width, slot)
}

/// [`latent_token`] of the page at byte offset `base`.
pub fn latent_token_at(
    pool: &[u8],
    base: usize,
    page: usize,
    width: usize,
    slot: usize,
) -> LabResult<(Vec<f64>, Vec<f32>)> {
    let codes = pool
        .get(base + slot * width..)
        .and_then(|bytes| bytes.get(..width))
        .ok_or_else(|| format!("byte {base} slot {slot} lies past the pool"))?;
    let blocks = width / BLOCK;
    let scales = pool
        .get(base + page * width + slot * blocks * 4..)
        .and_then(|bytes| bytes.get(..blocks * 4))
        .ok_or_else(|| format!("byte {base} slot {slot}'s scales lie past the pool"))?
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("4 bytes")))
        .collect();
    Ok((fp8_values(codes), scales))
}

/// `value` as the F32 a numerator holds, refused unless exact.
fn held(value: f64) -> LabResult<f32> {
    let held = value as f32;
    if f64::from(held) != value || !held.is_finite() {
        return Err(format!(
            "value {value} is not exactly a finite F32, so the numerator has no exact bits"
        ));
    }
    Ok(held)
}

/// One row's one-hot partials over `keys` kept keys: `dot(query, key)` is a
/// key's exact dot, scored at `score_scale`, and `value(key)` its row as the
/// numerator holds it. Without a kept key, -inf, +0 and +0.
fn one_hot(
    queries: &[f64],
    heads: usize,
    width: usize,
    keys: usize,
    score_scale: f32,
    dot: impl Fn(&[f64], usize) -> LabResult<f64>,
    value: impl Fn(usize) -> LabResult<Vec<f32>>,
) -> LabResult<Partials> {
    let mut partials = Partials {
        max: vec![f32::NEG_INFINITY; heads],
        den: vec![0.0; heads],
        num: vec![0.0; heads * width],
    };
    // A kernel weighs every kept value row, the losers by +0, which makes
    // NaN of an infinite element.
    let values = (0..keys).map(&value).collect::<LabResult<Vec<_>>>()?;
    for head in 0..heads {
        let query = &queries[head * width..][..width];
        let dots = (0..keys)
            .map(|key| dot(query, key))
            .collect::<LabResult<Vec<_>>>()?;
        let Some(best) = hot(&dots, f64::from(score_scale))? else {
            continue;
        };
        partials.max[head] = score(dots[best], score_scale);
        partials.den[head] = 1.0;
        partials.num[head * width..][..width].copy_from_slice(&values[best]);
    }
    Ok(partials)
}

/// The partials of a paged or selected partial entry's `heads` query rows
/// of `width` (`queries`, decoded) over the kept keys `(page id, slot)` of
/// the E4M3 K and V pools at `stride`: the score is the exact dot times the
/// F32 `k_scale * scale` and the numerator the best value row times
/// `v_scale`, each one F32 rounding as in the kernels.
#[allow(clippy::too_many_arguments)]
pub fn partials(
    queries: &[f64],
    heads: usize,
    width: usize,
    (k, v): (&[u8], &[u8]),
    stride: usize,
    keys: &[(i32, usize)],
    scale: f32,
    (k_scale, v_scale): (f32, f32),
) -> LabResult<Partials> {
    let rows = keys
        .iter()
        .map(|&(id, slot)| Ok(fp8_values(token(k, stride, width, id, slot)?)))
        .collect::<LabResult<Vec<_>>>()?;
    one_hot(
        queries,
        heads,
        width,
        keys.len(),
        k_scale * scale,
        |query, key| dot_values(query, &rows[key]),
        |key| {
            let (id, slot) = keys[key];
            fp8_values(token(v, stride, width, id, slot)?)
                .into_iter()
                .map(|value| {
                    let scaled = held(value)? * v_scale;
                    if !scaled.is_finite() {
                        return Err(format!("page {id} slot {slot}'s value row is not finite"));
                    }
                    Ok(scaled)
                })
                .collect()
        },
    )
}

/// The partials of a latent entry's `heads` query rows of `width` over the
/// kept keys `(page id, slot)` of an FP8 latent pool ([`latent_pool`]):
/// each 128-wide block's exact chain times the token's block scale, summed
/// in any order (a dot of the chains with the scales, which must be exact),
/// then times `scale`; the numerator is the best token's codes times their
/// block scales. Refused unless every block scale is a power of two, which
/// a kernel's BF16 probability terms then carry exactly, and unless every
/// dim's scaled values summed in magnitude over the kept keys stay within
/// half the largest F32: probabilities and rescales are at most 1, so that
/// sum bounds every running numerator, which then stays finite through its
/// F32 roundings even where a loser's term arrives before the winner.
#[allow(clippy::too_many_arguments)]
pub fn latent_partials(
    queries: &[f64],
    heads: usize,
    width: usize,
    pool: &[u8],
    stride: usize,
    page: usize,
    keys: &[(i32, usize)],
    scale: f32,
) -> LabResult<Partials> {
    let tokens = keys
        .iter()
        .map(|&(id, slot)| {
            let (codes, scales) = latent_token(pool, stride, page, width, id, slot)?;
            if let Some(bad) = scales
                .iter()
                .find(|s| !(s.is_normal() && **s > 0.0 && s.to_bits() & 0x007f_ffff == 0))
            {
                return Err(format!(
                    "page {id} slot {slot}'s block scale {bad} is not a power of two"
                ));
            }
            Ok((codes, scales))
        })
        .collect::<LabResult<Vec<_>>>()?;
    for d in 0..width {
        let sum = tokens
            .iter()
            .map(|(codes, scales)| (codes[d] * f64::from(scales[d / BLOCK])).abs())
            .sum::<f64>();
        if sum > f64::from(f32::MAX) / 2.0 {
            return Err(format!(
                "dim {d}'s scaled values sum to {sum} in magnitude, so a running numerator may \
                 overflow F32"
            ));
        }
    }
    one_hot(
        queries,
        heads,
        width,
        keys.len(),
        scale,
        |query, key| {
            let (codes, scales) = &tokens[key];
            let chains = (0..width / BLOCK)
                .map(|b| dot_values(&query[b * BLOCK..][..BLOCK], &codes[b * BLOCK..][..BLOCK]))
                .collect::<LabResult<Vec<_>>>()?;
            let scales = scales.iter().map(|&s| f64::from(s)).collect::<Vec<_>>();
            dot_values(&chains, &scales)
        },
        |key| {
            let (codes, scales) = &tokens[key];
            codes
                .iter()
                .enumerate()
                .map(|(d, &code)| held(code * f64::from(scales[d / BLOCK])))
                .collect()
        },
    )
}

/// `hot_keys::partial_outputs` over E4M3 K and V pools at [`K_SCALE`] and
/// [`V_SCALE`]: the exact partials `out_max`, `out_den` and `out_num` of a
/// paged or selected partial entry at `live` of `capacity` rows, reading
/// each table ID as `read` maps it. `kept(images, row, position)` lists the
/// positions row `row` attends.
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
    let q = crate::hot_keys::bf16_values(images.bytes("q")?);
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
            (K_SCALE, V_SCALE),
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
