use crate::{MediaError, MediaLimits, MediaResult, RgbImage};

use super::decode::{Component, Frame};
use super::idct;

struct Plane {
    pixels: Vec<u8>,
    stride: usize,
    width: usize,
    height: usize,
    h: usize,
    v: usize,
}

impl Plane {
    fn new(
        component: &Component,
        quant: &[u16; 64],
        max_h: usize,
        max_v: usize,
        image_width: usize,
        image_height: usize,
    ) -> MediaResult<Self> {
        let stride = component.block_width * 8;
        let rows = component.block_height * 8;
        let length = stride
            .checked_mul(rows)
            .ok_or_else(|| MediaError::new("JPEG sample plane overflow"))?;
        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(length)
            .map_err(|_| MediaError::new("JPEG sample plane allocation failed"))?;
        pixels.resize(length, 0);
        for by in 0..component.block_height {
            for bx in 0..component.block_width {
                let block = idct::block(&component.coeffs[by * component.block_width + bx], quant);
                for row in 0..8 {
                    let output = (by * 8 + row) * stride + bx * 8;
                    pixels[output..output + 8].copy_from_slice(&block[row * 8..row * 8 + 8]);
                }
            }
        }
        Ok(Self {
            pixels,
            stride,
            width: (image_width * component.h).div_ceil(max_h),
            height: (image_height * component.v).div_ceil(max_v),
            h: max_h / component.h,
            v: max_v / component.v,
        })
    }

    fn at(&self, x: usize, y: usize) -> i32 {
        i32::from(self.pixels[y.min(self.height - 1) * self.stride + x.min(self.width - 1)])
    }

    fn sample(&self, x: usize, y: usize) -> i32 {
        let sx = x / self.h;
        let sy = y / self.v;
        let fancy_h = self.h == 2 && self.width > 2;
        let fancy_v = self.v == 2 && (self.h != 2 || self.width > 2);
        if fancy_h && fancy_v {
            let vy = if y & 1 == 0 {
                sy.saturating_sub(1)
            } else {
                sy + 1
            };
            let sum = |column| 3 * self.at(column, sy) + self.at(column, vy);
            let current = sum(sx);
            if x & 1 == 0 {
                if sx == 0 {
                    (current * 4 + 8) >> 4
                } else {
                    (current * 3 + sum(sx - 1) + 8) >> 4
                }
            } else if sx + 1 >= self.width {
                (current * 4 + 7) >> 4
            } else {
                (current * 3 + sum(sx + 1) + 7) >> 4
            }
        } else if fancy_h {
            let current = self.at(sx, sy);
            if x & 1 == 0 {
                if sx == 0 {
                    current
                } else {
                    (3 * current + self.at(sx - 1, sy) + 1) >> 2
                }
            } else if sx + 1 >= self.width {
                current
            } else {
                (3 * current + self.at(sx + 1, sy) + 2) >> 2
            }
        } else if fancy_v {
            let vy = if y & 1 == 0 {
                sy.saturating_sub(1)
            } else {
                sy + 1
            };
            (3 * self.at(sx, sy) + self.at(sx, vy) + if y & 1 == 0 { 1 } else { 2 }) >> 2
        } else {
            self.at(sx, sy)
        }
    }
}

fn ycbcr(y: i32, cb: i32, cr: i32) -> [u8; 3] {
    let cb = cb - 128;
    let cr = cr - 128;
    let red = y + ((91881 * cr + 32768) >> 16);
    let green = y + ((-22554 * cb - 46802 * cr + 32768) >> 16);
    let blue = y + ((116130 * cb + 32768) >> 16);
    [
        red.clamp(0, 255) as u8,
        green.clamp(0, 255) as u8,
        blue.clamp(0, 255) as u8,
    ]
}

fn pillow_cmyk(raw: [i32; 4]) -> [u8; 3] {
    // Pillow's CMYK;I decoder always assumes Adobe polarity, including JPEGs
    // without APP14. Match its integer MULDIV255 conversion after inversion.
    let black = raw[3];
    std::array::from_fn(|index| {
        let product = (255 - raw[index]) * black + 128;
        let scaled = ((product >> 8) + product) >> 8;
        (black - scaled) as u8
    })
}

fn color(frame: &Frame, samples: &[i32; 4]) -> MediaResult<[u8; 3]> {
    match frame.components.len() {
        1 => Ok([samples[0] as u8; 3]),
        3 if frame.adobe_transform == Some(0)
            || frame.components.iter().map(|c| c.id).eq([b'R', b'G', b'B']) =>
        {
            Ok([samples[0] as u8, samples[1] as u8, samples[2] as u8])
        }
        3 if frame.adobe_transform.is_none_or(|transform| transform == 1) => {
            Ok(ycbcr(samples[0], samples[1], samples[2]))
        }
        4 if frame.adobe_transform == Some(2) => {
            let ycc_rgb = ycbcr(samples[0], samples[1], samples[2]);
            Ok(pillow_cmyk([
                255 - i32::from(ycc_rgb[0]),
                255 - i32::from(ycc_rgb[1]),
                255 - i32::from(ycc_rgb[2]),
                samples[3],
            ]))
        }
        4 if frame.adobe_transform.is_none_or(|transform| transform == 0) => {
            Ok(pillow_cmyk(*samples))
        }
        _ => Err(MediaError::new("unsupported JPEG color transform")),
    }
}

fn source_position(
    orientation: u8,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> (usize, usize) {
    match orientation {
        2 => (width - 1 - x, y),
        3 => (width - 1 - x, height - 1 - y),
        4 => (x, height - 1 - y),
        5 => (y, x),
        6 => (y, height - 1 - x),
        7 => (width - 1 - y, height - 1 - x),
        8 => (width - 1 - y, x),
        _ => (x, y),
    }
}

pub(super) fn render(frame: Frame, limits: MediaLimits) -> MediaResult<RgbImage> {
    let width = usize::from(frame.header.width);
    let height = usize::from(frame.header.height);
    let (out_width, out_height) = frame.header.oriented_size();
    limits.check_image(out_width, out_height)?;
    let mut planes = Vec::with_capacity(frame.components.len());
    for component in &frame.components {
        let quant = frame.quant[component.quant]
            .as_ref()
            .ok_or_else(|| MediaError::new("JPEG quantizer is missing"))?;
        planes.push(Plane::new(
            component,
            quant,
            frame.max_h,
            frame.max_v,
            width,
            height,
        )?);
    }
    let byte_count = width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| MediaError::new("JPEG RGB byte count overflow"))?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(byte_count)
        .map_err(|_| MediaError::new("JPEG RGB allocation failed"))?;
    for y in 0..out_height as usize {
        for x in 0..out_width as usize {
            let (sx, sy) = source_position(frame.header.orientation, x, y, width, height);
            let mut samples = [0_i32; 4];
            for (index, plane) in planes.iter().enumerate() {
                samples[index] = plane.sample(sx, sy);
            }
            pixels.extend_from_slice(&color(&frame, &samples)?);
        }
    }
    RgbImage::new(out_width, out_height, pixels, limits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_maps_all_eight_exif_values() {
        let expected = [
            (0, 0),
            (2, 0),
            (2, 1),
            (0, 1),
            (0, 0),
            (0, 1),
            (2, 1),
            (2, 0),
        ];
        for (index, position) in expected.iter().enumerate() {
            assert_eq!(source_position(index as u8 + 1, 0, 0, 3, 2), *position);
        }
    }

    #[test]
    fn ycbcr_uses_libjpeg_fixed_point_tables() {
        assert_eq!(ycbcr(128, 128, 128), [128; 3]);
        assert_eq!(ycbcr(76, 85, 255), [254, 0, 0]);
    }

    #[test]
    fn cmyk_uses_pillow_adobe_polarity_and_rounding() {
        assert_eq!(pillow_cmyk([128; 4]), [64; 3]);
        assert_eq!(pillow_cmyk([255; 4]), [255; 3]);
        assert_eq!(pillow_cmyk([0, 255, 128, 128]), [0, 128, 64]);
    }
}
