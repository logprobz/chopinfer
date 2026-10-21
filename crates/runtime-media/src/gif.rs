//! GIF decoding of the first frame as Pillow 12.3 reads it, over the `gif`
//! crate. The canvas is the logical screen grown to cover frame 0; pixels
//! outside the frame take the transparency index, else index 0 (never the
//! background index); a local palette wins over the global one. A palette
//! that maps every index to its own gray level reads as gray levels, so an
//! index past its end stays gray, while an index past any other palette is
//! black. Later frames are never decoded.

use std::io::Cursor;
use std::num::NonZeroU64;

use crate::core::try_with_capacity;
use crate::{Alpha, MediaError, MediaLimits, MediaResult, RgbImage};

/// The logical screen and frame 0's left, top, width and height.
#[derive(Clone, Copy)]
struct Layout {
    screen: (u32, u32),
    frame: (u32, u32, u32, u32),
}

impl Layout {
    fn canvas(self) -> (u32, u32) {
        let (left, top, width, height) = self.frame;
        (
            self.screen.0.max(left + width),
            self.screen.1.max(top + height),
        )
    }
}

fn color_table_bytes(flags: u8) -> usize {
    if flags & 0x80 != 0 {
        3 << ((flags & 7) + 1)
    } else {
        0
    }
}

/// Walks to frame 0 as Pillow does, skipping bytes that start no block, and
/// requires every sub-block before it and of its image data to be present,
/// so the crate decodes nothing the headers did not describe.
fn layout(bytes: &[u8]) -> MediaResult<Layout> {
    if bytes.len() < 13 || !(bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a")) {
        return Err(MediaError::new("GIF header is missing or truncated"));
    }
    let truncated = || MediaError::new("GIF is truncated before the end of its first frame");
    let get = |index: usize| bytes.get(index).copied().ok_or_else(truncated);
    let le16 = |index: usize| u32::from(u16::from_le_bytes([bytes[index], bytes[index + 1]]));
    let sub_blocks = |mut pos: usize| -> MediaResult<usize> {
        loop {
            let size = usize::from(get(pos)?);
            pos += 1 + size;
            if size == 0 {
                return Ok(pos);
            }
            if pos > bytes.len() {
                return Err(truncated());
            }
        }
    };
    let screen = (le16(6), le16(8));
    let mut pos = 13 + color_table_bytes(bytes[10]);
    loop {
        match get(pos)? {
            0x21 => pos = sub_blocks(pos + 2)?,
            0x2c => {
                let flags = get(pos + 9)?;
                let frame = (le16(pos + 1), le16(pos + 3), le16(pos + 5), le16(pos + 7));
                if frame.2 == 0 || frame.3 == 0 {
                    return Err(MediaError::new("GIF first frame has no pixels"));
                }
                let data = pos + 10 + color_table_bytes(flags);
                get(data)?;
                sub_blocks(data + 1)?;
                return Ok(Layout { screen, frame });
            }
            0x3b => return Err(MediaError::new("GIF has no image")),
            _ => pos += 1,
        }
    }
}

fn checked_layout(bytes: &[u8], limits: MediaLimits) -> MediaResult<Layout> {
    limits.check_bytes(bytes.len())?;
    let layout = layout(bytes)?;
    let (width, height) = layout.canvas();
    limits.check_named_image("GIF", width, height)?;
    Ok(layout)
}

/// The canvas size from the headers, checked against `limits`.
pub fn probe(bytes: &[u8], limits: MediaLimits) -> MediaResult<(u32, u32)> {
    Ok(checked_layout(bytes, limits)?.canvas())
}

/// Decoder storage derived from the headers, before any allocation: frame
/// 0's indices and the RGB canvas, plus the source as a bound on the
/// metadata extensions the crate keeps, and a fixed allowance for its
/// buffered reader, LZW tables and palettes.
pub fn decode_storage_bytes(bytes: &[u8], limits: MediaLimits) -> MediaResult<u64> {
    let layout = checked_layout(bytes, limits)?;
    let (width, height) = layout.canvas();
    let (_, _, frame_width, frame_height) = layout.frame;
    Ok(u64::from(frame_width) * u64::from(frame_height)
        + u64::from(width) * u64::from(height) * 3
        + bytes.len() as u64
        + 64 * 1024)
}

fn invalid(error: ::gif::DecodingError) -> MediaError {
    MediaError::new(format!("GIF image data is invalid: {error}"))
}

/// Decodes frame 0 onto its canvas. [`Alpha::Discard`] keeps the palette
/// color of the transparency index, as Pillow's `convert("RGB")` does;
/// [`Alpha::CompositeWhite`] makes it white.
pub fn decode(bytes: &[u8], limits: MediaLimits, alpha: Alpha) -> MediaResult<RgbImage> {
    let layout = checked_layout(bytes, limits)?;
    let (width, height) = layout.canvas();
    let (left, top, frame_width, frame_height) = layout.frame;
    let frame_pixels = u64::from(frame_width) * u64::from(frame_height);
    let mut options = ::gif::DecodeOptions::new();
    options.allow_unknown_blocks(true);
    options.set_color_output(::gif::ColorOutput::Indexed);
    // The crate also keeps XMP and ICC application extensions within this
    // limit, at most the source, as the storage bound counts them.
    options.set_memory_limit(::gif::MemoryLimit::Bytes(
        NonZeroU64::new(frame_pixels + bytes.len() as u64).expect("frame 0 has pixels"),
    ));
    let mut decoder = options.read_info(Cursor::new(bytes)).map_err(invalid)?;
    let global = decoder.global_palette().map(<[u8]>::to_vec);
    let frame = decoder
        .read_next_frame()
        .map_err(invalid)?
        .ok_or_else(|| MediaError::new("GIF has no image"))?;
    if (
        u32::from(frame.left),
        u32::from(frame.top),
        u32::from(frame.width),
        u32::from(frame.height),
    ) != layout.frame
        || frame.buffer.len() as u64 != frame_pixels
    {
        return Err(MediaError::new(
            "GIF first frame differs from its descriptor",
        ));
    }
    let palette = frame.palette.as_deref().or(global.as_deref());
    let gray = palette.is_none_or(|palette| {
        palette
            .chunks_exact(3)
            .enumerate()
            .all(|(index, rgb)| rgb.iter().all(|&value| usize::from(value) == index))
    });
    let colors: [[u8; 3]; 256] = std::array::from_fn(|index| {
        if alpha == Alpha::CompositeWhite && frame.transparent == Some(index as u8) {
            [255; 3]
        } else if gray {
            [index as u8; 3]
        } else {
            palette
                .and_then(|palette| palette.get(index * 3..index * 3 + 3))
                .map_or([0; 3], |rgb| [rgb[0], rgb[1], rgb[2]])
        }
    });
    let pixels = u64::from(width) * u64::from(height);
    let mut rgb = try_with_capacity(
        usize::try_from(pixels * 3).map_err(|_| MediaError::new("GIF pixel size overflow"))?,
    )?;
    let fill = colors[usize::from(frame.transparent.unwrap_or(0))];
    for _ in 0..pixels {
        rgb.extend_from_slice(&fill);
    }
    let (canvas_width, frame_width) = (width as usize, frame_width as usize);
    for (y, row) in frame.buffer.chunks_exact(frame_width).enumerate() {
        let start = ((top as usize + y) * canvas_width + left as usize) * 3;
        for (target, &index) in rgb[start..start + frame_width * 3]
            .chunks_exact_mut(3)
            .zip(row)
        {
            target.copy_from_slice(&colors[usize::from(index)]);
        }
    }
    RgbImage::new(width, height, rgb, limits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> MediaLimits {
        MediaLimits {
            maximum_bytes: 1 << 20,
            maximum_pixels: 1 << 20,
            maximum_audio_frames: 0,
            maximum_frames: 1,
            maximum_tokens: 0,
            maximum_aspect_ratio: 200.0,
        }
    }

    /// A frame's left, top, width, height, local palette, transparency
    /// index and indices.
    type Frame<'a> = (u16, u16, u16, u16, Option<&'a [u8]>, Option<u8>, &'a [u8]);

    /// A GIF with `screen`, an optional global palette, and `frames`.
    fn gif(screen: (u16, u16), global: Option<&[u8]>, frames: &[Frame<'_>]) -> Vec<u8> {
        let mut encoder =
            ::gif::Encoder::new(Vec::new(), screen.0, screen.1, global.unwrap_or(&[])).unwrap();
        for &(left, top, width, height, local, transparent, indices) in frames {
            let frame = ::gif::Frame {
                left,
                top,
                width,
                height,
                transparent,
                palette: local.map(<[u8]>::to_vec),
                buffer: indices.into(),
                ..Default::default()
            };
            encoder.write_frame(&frame).unwrap();
        }
        encoder.into_inner().unwrap()
    }

    const PALETTE: &[u8] = &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];

    #[test]
    fn frame_zero_grows_the_screen_and_fills_with_the_transparency_index() {
        // A 2 x 1 screen; frame 0 is 1 x 2 at (2, 0) with index 1
        // transparent, so the canvas is 3 x 2 filled with index 1.
        let bytes = gif(
            (2, 1),
            Some(PALETTE),
            &[
                (2, 0, 1, 2, None, Some(1), &[2, 3]),
                (0, 0, 1, 1, None, None, &[0]),
            ],
        );
        assert_eq!(probe(&bytes, limits()).unwrap(), (3, 2));
        let image = decode(&bytes, limits(), Alpha::Discard).unwrap();
        let (one, two, three) = ([40, 50, 60], [70, 80, 90], [100, 110, 120]);
        assert_eq!(
            image.data,
            [one, one, two, one, one, three].concat(),
            "the second frame is never drawn"
        );
        let white = decode(&bytes, limits(), Alpha::CompositeWhite).unwrap();
        assert_eq!(white.data[..3], [255; 3]);
        assert_eq!(white.data[6..9], two);
    }

    #[test]
    fn local_palette_wins_and_out_of_range_indices_are_black() {
        let local: &[u8] = &[1, 2, 3, 4, 5, 6];
        let bytes = gif(
            (3, 1),
            Some(PALETTE),
            &[(0, 0, 3, 1, Some(local), None, &[1, 0, 3])],
        );
        let image = decode(&bytes, limits(), Alpha::Discard).unwrap();
        assert_eq!(image.data, [4, 5, 6, 1, 2, 3, 0, 0, 0]);
        // An identity gray palette reads as gray levels, past its end too.
        let gray: &[u8] = &[0, 0, 0, 1, 1, 1];
        let bytes = gif((2, 1), Some(gray), &[(0, 0, 2, 1, None, None, &[1, 3])]);
        let image = decode(&bytes, limits(), Alpha::Discard).unwrap();
        assert_eq!(image.data, [1, 1, 1, 3, 3, 3]);
    }

    #[test]
    fn truncation_bombs_and_missing_frames_are_refused() {
        let bytes = gif(
            (2, 2),
            Some(PALETTE),
            &[(0, 0, 2, 2, None, None, &[0, 1, 2, 3])],
        );
        assert!(decode(&bytes, limits(), Alpha::Discard).is_ok());
        for cut in 0..bytes.len() - 1 {
            assert!(
                decode(&bytes[..cut], limits(), Alpha::Discard).is_err(),
                "{cut}"
            );
        }
        let mut bomb = bytes.clone();
        bomb[6..10].copy_from_slice(&[0xff; 4]);
        assert_eq!(
            probe(&bomb, limits()).unwrap_err().to_string(),
            "GIF image 65535 x 65535 has 4294836225 pixels, above the 1048576-pixel limit"
        );
        let empty = [&bytes[..13 + PALETTE.len()], b";"].concat();
        assert!(decode(&empty, limits(), Alpha::Discard).is_err());
    }
}
