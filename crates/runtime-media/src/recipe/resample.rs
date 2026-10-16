//! Separable antialiased bicubic resize on 8-bit samples.
//!
//! This is the integer path of Pillow's `ImagingResample`, as PyTorch's CPU
//! `_upsample_bicubic2d_aa` runs it on uint8 tensors: per-axis filter weights
//! computed in f64 with the Keys cubic (a = -0.5) stretched by the downscale
//! factor, normalized per output sample, then quantized to 16-bit fixed point
//! at the largest precision that keeps the biggest weight below 2^15. The
//! horizontal pass runs first into an 8-bit buffer, then the vertical pass.
//! Each output sample is `clamp((2^(p-1) + sum(in * w)) >> p, 0, 255)`.
use std::borrow::Cow;

use crate::core::try_filled;
use crate::{MediaResult, RgbImage};

struct AxisWeights {
    /// First input index and fixed-point weights for each output index.
    taps: Vec<(usize, Vec<i32>)>,
    precision: u32,
}

fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

fn axis_weights(input: usize, output: usize) -> AxisWeights {
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
            .map(|j| cubic(((j + first) as f64 - center + 0.5) * inverse))
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
    let mut precision = 0;
    while precision < 22 && ((0.5 + largest * f64::from(1u32 << (precision + 1))) as i32) < 1 << 15
    {
        precision += 1;
    }
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

fn fixed_point(sum: i32, precision: u32) -> u8 {
    (sum >> precision).clamp(0, 255) as u8
}

/// Resize interleaved RGB to `width` by `height`, returning interleaved RGB.
pub(crate) fn bicubic_antialias(
    image: &RgbImage,
    height: u32,
    width: u32,
) -> MediaResult<Cow<'_, [u8]>> {
    let (in_w, in_h) = (image.width as usize, image.height as usize);
    let (out_w, out_h) = (width as usize, height as usize);
    let horizontal = if out_w == in_w {
        Cow::Borrowed(image.data.as_slice())
    } else {
        let weights = axis_weights(in_w, out_w);
        let round = 1i32 << (weights.precision - 1);
        let mut out = try_filled(in_h * out_w * 3, 0u8)?;
        for (source, row) in image
            .data
            .chunks_exact(in_w * 3)
            .zip(out.chunks_exact_mut(out_w * 3))
        {
            for ((first, taps), pixel) in weights.taps.iter().zip(row.chunks_exact_mut(3)) {
                for (c, sample) in pixel.iter_mut().enumerate() {
                    let sum = taps.iter().enumerate().fold(round, |sum, (k, &w)| {
                        sum + i32::from(source[(first + k) * 3 + c]) * w
                    });
                    *sample = fixed_point(sum, weights.precision);
                }
            }
        }
        Cow::Owned(out)
    };
    if out_h == in_h {
        return Ok(horizontal);
    }
    let weights = axis_weights(in_h, out_h);
    let round = 1i32 << (weights.precision - 1);
    let line = out_w * 3;
    let mut out = try_filled(out_h * line, 0u8)?;
    for ((first, taps), row) in weights.taps.iter().zip(out.chunks_exact_mut(line)) {
        for (x, sample) in row.iter_mut().enumerate() {
            let sum = taps.iter().enumerate().fold(round, |sum, (k, &w)| {
                sum + i32::from(horizontal[(first + k) * line + x]) * w
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
        let w = axis_weights(8, 4);
        assert_eq!(w.precision, 16);
        assert_eq!(w.taps[0], (0, vec![30437, 30437, 7952, -2468, -823]));
        assert_eq!(
            w.taps[1],
            (0, vec![-2277, 7338, 28087, 28087, 7338, -2277, -759])
        );
        assert_eq!(w.taps[3], (3, vec![-823, -2468, 7952, 30437, 30437]));
        // Doubling uses the unstretched kernel at quarter-pixel offsets.
        let w = axis_weights(2, 4);
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
}
