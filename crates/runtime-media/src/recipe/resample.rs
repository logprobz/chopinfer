//! Separable antialiased bicubic resize on 8-bit samples, in two weight
//! precisions.
//!
//! Both are the integer path of Pillow's `ImagingResample`: per-axis filter
//! weights computed in f64 with the Keys cubic (a = -0.5) stretched by the
//! downscale factor, normalized per output sample, then quantized to fixed
//! point at precision `p`. The horizontal pass runs first into an 8-bit
//! buffer, then the vertical pass. Each output sample is
//! `clamp((2^(p-1) + sum(in * w)) >> p, 0, 255)`.
//!
//! - [`bicubic_antialias`] is the path PyTorch's CPU
//!   `_upsample_bicubic2d_aa` runs on uint8 tensors: `p` is the largest
//!   precision that keeps the biggest weight below 2^15, and the cubic's
//!   outer lobe is evaluated as `((a x - 5 a) x + 8 a) x - 4 a`.
//! - [`bicubic_pillow`] is Pillow's own (`Image.resize` with `BICUBIC`):
//!   `p` is always 22 (`PRECISION_BITS`, 32 - 8 - 2), the outer lobe is
//!   evaluated as `(((x - 5) x + 8) x - 4) a`, and, as Pillow 12.3's
//!   `Image.resize` does, a source more than 100 times taller than wide
//!   whose height shrinks takes the vertical pass first.
use std::borrow::Cow;

use crate::core::try_filled;
use crate::{MediaResult, RgbImage};

struct AxisWeights {
    /// First input index and fixed-point weights for each output index.
    taps: Vec<(usize, Vec<i32>)>,
    precision: u32,
}

/// The two fixed-point conventions of the 8-bit path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fixed {
    /// PyTorch's: the largest precision keeping every weight below 2^15.
    Adaptive,
    /// Pillow's: 22 bits.
    Pillow,
}

/// Pillow's `PRECISION_BITS`.
const PILLOW_PRECISION: u32 = 22;

fn cubic(x: f64, fixed: Fixed) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        match fixed {
            Fixed::Adaptive => ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A,
            Fixed::Pillow => (((x - 5.0) * x + 8.0) * x - 4.0) * A,
        }
    } else {
        0.0
    }
}

fn axis_weights(input: usize, output: usize, fixed: Fixed) -> AxisWeights {
    let scale = input as f64 / output as f64;
    let (support, inverse) = if scale >= 1.0 {
        (2.0 * scale, 1.0 / scale)
    } else {
        (2.0, 1.0)
    };
    let maximum_taps = support.ceil() as i64 * 2 + 1;
    let mut largest = 0.0f64;
    let mut real = Vec::with_capacity(output);
    for i in 0..output {
        let center = scale * (i as f64 + 0.5);
        let first = ((center - support + 0.5) as i64).max(0);
        let count =
            (((center + support + 0.5) as i64).min(input as i64) - first).clamp(0, maximum_taps);
        let mut weights: Vec<f64> = (0..count)
            .map(|j| cubic(((j + first) as f64 - center + 0.5) * inverse, fixed))
            .collect();
        let total = weights.iter().fold(0.0, |sum, &w| sum + w);
        if total != 0.0 {
            for weight in &mut weights {
                *weight /= total;
                largest = largest.max(*weight);
            }
        }
        real.push((first as usize, weights));
    }
    let precision = match fixed {
        Fixed::Pillow => PILLOW_PRECISION,
        Fixed::Adaptive => {
            let mut precision = 0;
            while precision < 22
                && ((0.5 + largest * f64::from(1u32 << (precision + 1))) as i32) < 1 << 15
            {
                precision += 1;
            }
            precision
        }
    };
    let taps = real
        .into_iter()
        .map(|(first, weights)| {
            let weights = weights
                .into_iter()
                .map(|w| {
                    let v = w * f64::from(1u32 << precision);
                    (if v < 0.0 { v - 0.5 } else { v + 0.5 }) as i32
                })
                .collect();
            (first, weights)
        })
        .collect();
    AxisWeights { taps, precision }
}

fn fixed_point(sum: i64, precision: u32) -> u8 {
    (sum >> precision).clamp(0, 255) as u8
}

/// Resize interleaved RGB to `width` by `height` with PyTorch's 16-bit
/// weights, returning interleaved RGB.
pub(crate) fn bicubic_antialias(
    image: &RgbImage,
    height: u32,
    width: u32,
) -> MediaResult<Cow<'_, [u8]>> {
    resize(image, height, width, Fixed::Adaptive)
}

/// Resize interleaved RGB to `width` by `height` with Pillow's 22-bit
/// weights, returning interleaved RGB.
pub(crate) fn bicubic_pillow(
    image: &RgbImage,
    height: u32,
    width: u32,
) -> MediaResult<Cow<'_, [u8]>> {
    resize(image, height, width, Fixed::Pillow)
}

fn resize(image: &RgbImage, height: u32, width: u32, fixed: Fixed) -> MediaResult<Cow<'_, [u8]>> {
    let (in_w, in_h) = (image.width as usize, image.height as usize);
    let (out_w, out_h) = (width as usize, height as usize);
    let source = Cow::Borrowed(image.data.as_slice());
    // Pillow's `Image.resize` shrinks a source more than 100 times taller
    // than wide vertically first, as two one-axis resizes.
    if fixed == Fixed::Pillow && in_h > in_w * 100 && out_h < in_h {
        let vertical = vertical_pass(source, in_w, in_h, out_h, fixed)?;
        return horizontal_pass(vertical, in_w, out_h, out_w, fixed);
    }
    let horizontal = horizontal_pass(source, in_w, in_h, out_w, fixed)?;
    vertical_pass(horizontal, out_w, in_h, out_h, fixed)
}

/// Resizes interleaved RGB rows of `in_w` pixels to `out_w`.
fn horizontal_pass(
    source: Cow<'_, [u8]>,
    in_w: usize,
    rows: usize,
    out_w: usize,
    fixed: Fixed,
) -> MediaResult<Cow<'_, [u8]>> {
    if out_w == in_w {
        return Ok(source);
    }
    let weights = axis_weights(in_w, out_w, fixed);
    let round = 1i64 << (weights.precision - 1);
    let mut out = try_filled(rows * out_w * 3, 0u8)?;
    for (source, row) in source
        .chunks_exact(in_w * 3)
        .zip(out.chunks_exact_mut(out_w * 3))
    {
        for ((first, taps), pixel) in weights.taps.iter().zip(row.chunks_exact_mut(3)) {
            for (c, sample) in pixel.iter_mut().enumerate() {
                let sum = taps.iter().enumerate().fold(round, |sum, (k, &w)| {
                    sum + i64::from(source[(first + k) * 3 + c]) * i64::from(w)
                });
                *sample = fixed_point(sum, weights.precision);
            }
        }
    }
    Ok(Cow::Owned(out))
}

/// Resizes interleaved RGB columns of `in_h` pixels in rows of `width` to
/// `out_h`.
fn vertical_pass(
    source: Cow<'_, [u8]>,
    width: usize,
    in_h: usize,
    out_h: usize,
    fixed: Fixed,
) -> MediaResult<Cow<'_, [u8]>> {
    if out_h == in_h {
        return Ok(source);
    }
    let weights = axis_weights(in_h, out_h, fixed);
    let round = 1i64 << (weights.precision - 1);
    let line = width * 3;
    let mut out = try_filled(out_h * line, 0u8)?;
    for ((first, taps), row) in weights.taps.iter().zip(out.chunks_exact_mut(line)) {
        for (x, sample) in row.iter_mut().enumerate() {
            let sum = taps.iter().enumerate().fold(round, |sum, (k, &w)| {
                sum + i64::from(source[(first + k) * line + x]) * i64::from(w)
            });
            *sample = fixed_point(sum, weights.precision);
        }
    }
    Ok(Cow::Owned(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32, data: Vec<u8>) -> RgbImage {
        RgbImage {
            width,
            height,
            data,
        }
    }

    #[test]
    fn identity_and_constant_images_are_preserved() {
        let data: Vec<u8> = (0..4 * 3 * 3).map(|v| (v * 7) as u8).collect();
        let same = image(4, 3, data.clone());
        assert_eq!(*bicubic_antialias(&same, 3, 4).unwrap(), data[..]);
        let flat = image(7, 5, [9, 128, 250].repeat(35));
        assert_eq!(
            *bicubic_antialias(&flat, 3, 2).unwrap(),
            [9, 128, 250].repeat(6)[..]
        );
        assert_eq!(
            *bicubic_antialias(&flat, 11, 13).unwrap(),
            [9, 128, 250].repeat(143)[..]
        );
    }

    #[test]
    fn weights_quantize_like_the_fixed_point_reference() {
        // Halving stretches the kernel to 4 taps each side; the edge rows
        // lose their out-of-range taps and are renormalized.
        let w = axis_weights(8, 4, Fixed::Adaptive);
        assert_eq!(w.precision, 16);
        assert_eq!(w.taps[0], (0, vec![30437, 30437, 7952, -2468, -823]));
        assert_eq!(
            w.taps[1],
            (0, vec![-2277, 7338, 28087, 28087, 7338, -2277, -759])
        );
        assert_eq!(w.taps[3], (3, vec![-823, -2468, 7952, 30437, 30437]));
        // Doubling uses the unstretched kernel at quarter-pixel offsets.
        let w = axis_weights(2, 4, Fixed::Adaptive);
        assert_eq!(w.precision, 14);
        assert_eq!(w.taps[0], (0, vec![17830, -1446]));
        assert_eq!(w.taps[1], (0, vec![12990, 3394]));
    }

    #[test]
    fn step_edge_overshoots_and_clamps() {
        // A 0/255 edge halved horizontally: the negative lobes push the
        // outer samples past both ends, where they clamp.
        let row = [0, 0, 0, 0, 255, 255, 255, 255];
        let data: Vec<u8> = row.iter().flat_map(|&v| [v, v, v]).collect();
        let edge = image(8, 2, data.repeat(2));
        let out = bicubic_antialias(&edge, 2, 4).unwrap();
        let red: Vec<u8> = out.chunks_exact(3).map(|p| p[0]).collect();
        assert_eq!(red, [0, 17, 238, 255, 0, 17, 238, 255]);
    }

    #[test]
    fn pillow_weights_use_22_bits_and_its_lobe_order() {
        let w = axis_weights(8, 4, Fixed::Pillow);
        assert_eq!(w.precision, 22);
        // The adaptive weights at 16 bits, scaled by 2^6, are within the
        // rounding of the 22-bit ones.
        let adaptive = axis_weights(8, 4, Fixed::Adaptive);
        for ((first, taps), (first16, taps16)) in w.taps.iter().zip(&adaptive.taps) {
            assert_eq!(first, first16);
            for (&t, &t16) in taps.iter().zip(taps16) {
                assert!((t - t16 * 64).abs() <= 32, "{t} against {t16}");
            }
        }
        // Constant images stay constant; identity sizes are unchanged.
        let flat = image(7, 5, [9, 128, 250].repeat(35));
        assert_eq!(
            *bicubic_pillow(&flat, 3, 2).unwrap(),
            [9, 128, 250].repeat(6)[..]
        );
        let data: Vec<u8> = (0..4 * 3 * 3).map(|v| (v * 7) as u8).collect();
        assert_eq!(
            *bicubic_pillow(&image(4, 3, data.clone()), 3, 4).unwrap(),
            data[..]
        );
    }
}
