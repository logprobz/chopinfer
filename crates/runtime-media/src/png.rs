//! Bounded PNG decoder. The inflate interface keeps compression separate from parsing.

use crate::{Alpha, MediaError, MediaLimits, MediaResult, RgbImage};

pub trait Inflate {
    fn zlib_exact(&self, compressed: &[u8], expected: usize) -> MediaResult<Vec<u8>>;
}

pub struct MinizInflate;

impl Inflate for MinizInflate {
    fn zlib_exact(&self, compressed: &[u8], expected: usize) -> MediaResult<Vec<u8>> {
        let bytes = miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(compressed, expected)
            .map_err(|_| MediaError::new("PNG inflate failed or exceeded decoded limit"))?;
        if bytes.len() != expected {
            return Err(MediaError::new("PNG inflated byte count differs"));
        }
        Ok(bytes)
    }
}

#[derive(Clone, Copy)]
struct Header {
    width: u32,
    height: u32,
    depth: u8,
    color: u8,
    interlace: u8,
}

impl Header {
    fn channels(self) -> usize {
        match self.color {
            0 | 3 => 1,
            2 => 3,
            4 => 2,
            6 => 4,
            _ => unreachable!(),
        }
    }

    fn row_bytes(self, width: usize) -> MediaResult<usize> {
        width
            .checked_mul(self.channels())
            .and_then(|n| n.checked_mul(self.depth as usize))
            .and_then(|n| n.checked_add(7))
            .map(|n| n / 8)
            .ok_or_else(|| MediaError::new("PNG row size overflow"))
    }
}

fn be32(bytes: &[u8]) -> u32 {
    u32::from_be_bytes(bytes.try_into().expect("four bytes"))
}

fn header(bytes: &[u8]) -> MediaResult<Header> {
    if bytes.len() < 33
        || !bytes.starts_with(b"\x89PNG\r\n\x1a\n")
        || be32(&bytes[8..12]) != 13
        || &bytes[12..16] != b"IHDR"
    {
        return Err(MediaError::new("PNG IHDR is missing or truncated"));
    }
    let h = Header {
        width: be32(&bytes[16..20]),
        height: be32(&bytes[20..24]),
        depth: bytes[24],
        color: bytes[25],
        interlace: bytes[28],
    };
    let valid_depth = match h.color {
        0 | 3 => matches!(h.depth, 1 | 2 | 4 | 8),
        2 | 4 | 6 => h.depth == 8,
        _ => false,
    };
    if h.width == 0
        || h.height == 0
        || !valid_depth
        || bytes[26] != 0
        || bytes[27] != 0
        || h.interlace > 1
    {
        return Err(MediaError::new("unsupported PNG IHDR"));
    }
    Ok(h)
}

pub fn probe(bytes: &[u8]) -> MediaResult<(u32, u32)> {
    let h = header(bytes)?;
    Ok((h.width, h.height))
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (0_u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

fn scanlines(h: Header) -> MediaResult<usize> {
    let mut total = 0_usize;
    for &(sx, sy, dx, dy) in passes(h.interlace) {
        let w = pass_size(h.width, sx, dx) as usize;
        let rows = pass_size(h.height, sy, dy) as usize;
        if w == 0 || rows == 0 {
            continue;
        }
        let stride = h
            .row_bytes(w)?
            .checked_add(1)
            .ok_or_else(|| MediaError::new("PNG scanline overflow"))?;
        total = total
            .checked_add(
                stride
                    .checked_mul(rows)
                    .ok_or_else(|| MediaError::new("PNG scanline overflow"))?,
            )
            .ok_or_else(|| MediaError::new("PNG scanline overflow"))?;
    }
    Ok(total)
}

fn pass_size(size: u32, start: u32, step: u32) -> u32 {
    if size <= start {
        0
    } else {
        1 + (size - 1 - start) / step
    }
}

fn passes(interlace: u8) -> &'static [(u32, u32, u32, u32)] {
    const PLAIN: &[(u32, u32, u32, u32)] = &[(0, 0, 1, 1)];
    const ADAM7: &[(u32, u32, u32, u32)] = &[
        (0, 0, 8, 8),
        (4, 0, 8, 8),
        (0, 4, 4, 8),
        (2, 0, 4, 4),
        (0, 2, 2, 4),
        (1, 0, 2, 2),
        (0, 1, 1, 2),
    ];
    if interlace == 0 { PLAIN } else { ADAM7 }
}

fn sample(line: &[u8], index: usize, depth: u8) -> u8 {
    if depth == 8 {
        return line[index];
    }
    let bit = index * depth as usize;
    let shift = 8 - depth as usize - bit % 8;
    (line[bit / 8] >> shift) & ((1 << depth) - 1)
}

fn composite(rgb: [u8; 3], alpha: u8, policy: Alpha) -> [u8; 3] {
    if policy == Alpha::Discard {
        return rgb;
    }
    rgb.map(|channel| {
        ((u16::from(channel) * u16::from(alpha) + 255 * (255 - u16::from(alpha)) + 127) / 255) as u8
    })
}

fn pixel(
    line: &[u8],
    x: usize,
    h: Header,
    palette: &[[u8; 3]],
    transparency: &[u8],
    policy: Alpha,
) -> MediaResult<[u8; 3]> {
    let value = match h.color {
        0 => {
            let raw = sample(line, x, h.depth);
            let gray = (u16::from(raw) * 255 / ((1 << h.depth) - 1)) as u8;
            let transparent = transparency.len() == 2
                && u16::from_be_bytes([transparency[0], transparency[1]]) == u16::from(raw);
            composite([gray; 3], if transparent { 0 } else { 255 }, policy)
        }
        2 => {
            let raw = &line[x * 3..x * 3 + 3];
            let transparent = transparency.len() == 6
                && raw.iter().enumerate().all(|(i, &c)| {
                    u16::from_be_bytes([transparency[i * 2], transparency[i * 2 + 1]])
                        == u16::from(c)
                });
            composite(
                raw.try_into().expect("RGB triplet"),
                if transparent { 0 } else { 255 },
                policy,
            )
        }
        3 => {
            let index = sample(line, x, h.depth) as usize;
            let rgb = *palette
                .get(index)
                .ok_or_else(|| MediaError::new("PNG palette index exceeds PLTE"))?;
            composite(rgb, transparency.get(index).copied().unwrap_or(255), policy)
        }
        4 => composite([line[x * 2]; 3], line[x * 2 + 1], policy),
        6 => composite(
            line[x * 4..x * 4 + 3].try_into().expect("RGBA triplet"),
            line[x * 4 + 3],
            policy,
        ),
        _ => unreachable!(),
    };
    Ok(value)
}

fn unfilter(filter: u8, row: &mut [u8], previous: &[u8], bpp: usize) -> MediaResult<()> {
    for i in 0..row.len() {
        let left = if i >= bpp { row[i - bpp] } else { 0 };
        let up = previous[i];
        let upper_left = if i >= bpp { previous[i - bpp] } else { 0 };
        let predictor = match filter {
            0 => 0,
            1 => left,
            2 => up,
            3 => ((u16::from(left) + u16::from(up)) / 2) as u8,
            4 => {
                let p = i32::from(left) + i32::from(up) - i32::from(upper_left);
                let a = (p - i32::from(left)).abs();
                let b = (p - i32::from(up)).abs();
                let c = (p - i32::from(upper_left)).abs();
                if a <= b && a <= c {
                    left
                } else if b <= c {
                    up
                } else {
                    upper_left
                }
            }
            _ => return Err(MediaError::new("unknown PNG row filter")),
        };
        row[i] = row[i].wrapping_add(predictor);
    }
    Ok(())
}

/// Decodes with transparency composited onto white.
pub fn decode(bytes: &[u8], limits: MediaLimits) -> MediaResult<RgbImage> {
    decode_with_alpha(bytes, limits, Alpha::CompositeWhite)
}

/// Decodes with transparency (an alpha channel or a tRNS chunk) handled
/// as `alpha` says.
pub fn decode_with_alpha(bytes: &[u8], limits: MediaLimits, alpha: Alpha) -> MediaResult<RgbImage> {
    decode_inner(bytes, limits, &MinizInflate, alpha)
}

fn decode_inner(
    bytes: &[u8],
    limits: MediaLimits,
    inflate: &impl Inflate,
    policy: Alpha,
) -> MediaResult<RgbImage> {
    limits.check_bytes(bytes.len())?;
    let h = header(bytes)?;
    limits.check_image(h.width, h.height)?;
    let mut pos = 8_usize;
    let mut palette = Vec::new();
    let mut transparency = Vec::new();
    let mut idat = Vec::new();
    let mut seen_ihdr = false;
    let mut seen_iend = false;
    let mut seen_idat = false;
    while pos + 12 <= bytes.len() {
        let len = be32(&bytes[pos..pos + 4]) as usize;
        let end = pos
            .checked_add(12)
            .and_then(|n| n.checked_add(len))
            .ok_or_else(|| MediaError::new("PNG chunk length overflow"))?;
        if end > bytes.len() {
            return Err(MediaError::new("truncated PNG chunk"));
        }
        let kind = &bytes[pos + 4..pos + 8];
        let data = &bytes[pos + 8..pos + 8 + len];
        if crc32(&bytes[pos + 4..pos + 8 + len]) != be32(&bytes[pos + 8 + len..end]) {
            return Err(MediaError::new("PNG chunk CRC differs"));
        }
        match kind {
            b"IHDR" if !seen_ihdr && pos == 8 && len == 13 => seen_ihdr = true,
            b"PLTE"
                if seen_ihdr && !seen_idat && len % 3 == 0 && len <= 768 && !data.is_empty() =>
            {
                palette = data
                    .chunks_exact(3)
                    .map(|rgb| [rgb[0], rgb[1], rgb[2]])
                    .collect();
            }
            b"tRNS" if seen_ihdr && !seen_idat => transparency = data.to_vec(),
            b"IDAT" if seen_ihdr => {
                seen_idat = true;
                idat.extend_from_slice(data);
                if idat.len() > limits.maximum_bytes {
                    return Err(MediaError::new("PNG IDAT byte limit exceeded"));
                }
            }
            b"IEND" if seen_ihdr && len == 0 => {
                seen_iend = true;
                pos = end;
                break;
            }
            _ if kind[0] & 0x20 != 0 => {}
            _ => return Err(MediaError::new("unsupported or misplaced PNG chunk")),
        }
        pos = end;
    }
    if !seen_iend
        || pos != bytes.len()
        || !seen_idat
        || (h.color == 3 && palette.is_empty())
        || (h.color == 3 && transparency.len() > palette.len())
        || (h.color == 0 && !matches!(transparency.len(), 0 | 2))
        || (h.color == 2 && !matches!(transparency.len(), 0 | 6))
        || (matches!(h.color, 4 | 6) && !transparency.is_empty())
    {
        return Err(MediaError::new(
            "PNG chunk set is incomplete or inconsistent",
        ));
    }
    let raw = inflate.zlib_exact(&idat, scanlines(h)?)?;
    let pixel_count = u64::from(h.width) * u64::from(h.height);
    let output_len = usize::try_from(
        pixel_count
            .checked_mul(3)
            .ok_or_else(|| MediaError::new("PNG pixel size overflow"))?,
    )
    .map_err(|_| MediaError::new("PNG pixel size overflow"))?;
    let mut rgb = crate::core::try_filled(output_len, 0_u8)?;
    let mut source = 0_usize;
    let bpp = ((h.channels() * h.depth as usize).div_ceil(8)).max(1);
    for &(sx, sy, dx, dy) in passes(h.interlace) {
        let width = pass_size(h.width, sx, dx) as usize;
        let height = pass_size(h.height, sy, dy) as usize;
        if width == 0 || height == 0 {
            continue;
        }
        let stride = h.row_bytes(width)?;
        let mut previous = vec![0_u8; stride];
        for y in 0..height {
            let filter = raw[source];
            source += 1;
            let mut row = raw[source..source + stride].to_vec();
            source += stride;
            unfilter(filter, &mut row, &previous, bpp)?;
            for x in 0..width {
                let target_x = sx as usize + x * dx as usize;
                let target_y = sy as usize + y * dy as usize;
                let target = (target_y * h.width as usize + target_x) * 3;
                rgb[target..target + 3].copy_from_slice(&pixel(
                    &row,
                    x,
                    h,
                    &palette,
                    &transparency,
                    policy,
                )?);
            }
            previous = row;
        }
    }
    RgbImage::new(h.width, h.height, rgb, limits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(kind: &[u8; 4], payload: &[u8], output: &mut Vec<u8>) {
        output.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        output.extend_from_slice(kind);
        output.extend_from_slice(payload);
        let start = output.len() - payload.len() - 4;
        output.extend_from_slice(&crc32(&output[start..]).to_be_bytes());
    }

    fn limits() -> MediaLimits {
        MediaLimits {
            maximum_bytes: 1 << 20,
            maximum_pixels: 1 << 20,
            maximum_audio_frames: 1 << 20,
            maximum_frames: 32,
            maximum_tokens: 1024,
            maximum_aspect_ratio: 200.0,
        }
    }

    #[test]
    fn adam7_palette_alpha_composites_on_white() {
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1_u32.to_be_bytes());
        ihdr.extend_from_slice(&1_u32.to_be_bytes());
        ihdr.extend_from_slice(&[1, 3, 0, 0, 1]);
        chunk(b"IHDR", &ihdr, &mut png);
        chunk(b"PLTE", &[0, 0, 255, 255, 0, 0], &mut png);
        chunk(b"tRNS", &[128, 255], &mut png);
        chunk(
            b"IDAT",
            &miniz_oxide::deflate::compress_to_vec_zlib(&[0, 0], 6),
            &mut png,
        );
        chunk(b"IEND", &[], &mut png);
        assert_eq!(decode(&png, limits()).unwrap().data, [127, 127, 255]);
        let discarded = decode_with_alpha(&png, limits(), Alpha::Discard).unwrap();
        assert_eq!(discarded.data, [0, 0, 255]);
        let mut bad = png;
        bad[29] ^= 1;
        assert!(decode(&bad, limits()).is_err());
    }
}
