//! WebP decoding as Pillow 12.3 reads it through libwebp's animation
//! decoder, over the `image-webp` crate behind a container guard. A still
//! image is its plain decode; an animated file is its first frame, written
//! unblended onto a transparent black canvas.
//!
//! The guard parses the RIFF chunks and the VP8X, ANMF, VP8 and VP8L headers
//! first. It refuses reserved VP8X flags (as libwebp's demuxer does), a
//! frame outside its canvas, a bitstream whose size differs from its canvas
//! or frame and a VP8 partition that runs past its chunk, then hands
//! image-webp one rebuilt still frame. image-webp decodes a lossy frame
//! before it compares sizes, allocates each VP8 partition's declared size
//! before reading it, and its memory limit covers only metadata chunks, so
//! it never sees anything else.

use std::io::Cursor;

use crate::core::{try_filled, try_with_capacity};
use crate::png::composite;
use crate::{Alpha, MediaError, MediaLimits, MediaResult, Orientation, RgbImage};

/// Host bytes allowed per byte of a lossless (VP8L or ALPH) bitstream. A
/// VP8L stream declares up to 65,536 meta-code groups of five Huffman codes,
/// and image-webp builds every one of them: a code of a few dozen bits can
/// fill a 4 KiB lookup table, and repeated code lengths add 32 bytes of tree
/// per symbol for about half a bit each. Neither depends on the image size,
/// so the bound counts them by the bytes that declare them, at about twice
/// the densest case.
const LOSSLESS_BYTES_PER_BYTE: u64 = 1024;

/// Host bytes allowed per byte of a lossy (VP8) bitstream: image-webp
/// copies every partition, reading the last one through a buffer that
/// doubles as it grows, then copies that into a buffer it shrinks by one
/// more copy.
const LOSSY_BYTES_PER_BYTE: u64 = 4;

/// Decoder bytes allowed per frame pixel, over the frame padded to whole
/// 16 x 16 macroblocks: the RGBA output, the lossy Y, U and V planes, a
/// decoded alpha plane and the lossless decoder's ARGB rows.
const FRAME_BYTES_PER_PIXEL: u64 = 16;

#[derive(Clone, Copy)]
struct Chunk<'a> {
    fourcc: [u8; 4],
    data: &'a [u8],
}

fn le24(bytes: &[u8]) -> u32 {
    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16)
}

fn riff_chunks(bytes: &[u8]) -> MediaResult<Vec<Chunk<'_>>> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return Err(MediaError::new("WebP RIFF header is missing or truncated"));
    }
    let declared = u32::from_le_bytes(bytes[4..8].try_into().expect("four bytes")) as usize;
    let end = declared
        .checked_add(8)
        .filter(|&end| (12..=bytes.len()).contains(&end))
        .ok_or_else(|| MediaError::new("WebP RIFF size exceeds the file"))?;
    sub_chunks(&bytes[12..end])
}

/// The chunks of a RIFF body; the last one's pad byte may be missing.
fn sub_chunks(mut body: &[u8]) -> MediaResult<Vec<Chunk<'_>>> {
    let mut chunks = Vec::new();
    while !body.is_empty() {
        if body.len() < 8 {
            return Err(MediaError::new("WebP chunk header is truncated"));
        }
        let size = u32::from_le_bytes(body[4..8].try_into().expect("four bytes")) as usize;
        if body.len() - 8 < size {
            return Err(MediaError::new("WebP chunk is truncated"));
        }
        chunks.push(Chunk {
            fourcc: body[..4].try_into().expect("four bytes"),
            data: &body[8..8 + size],
        });
        body = &body[(8 + size + (size & 1)).min(body.len())..];
    }
    Ok(chunks)
}

/// RFC 6386's boolean decoder at even probability, which codes a VP8 frame
/// header's fields, reading zeros past the end of its data. image-webp
/// refuses a header that reads more than a byte past its partition before
/// it reads the partitions, so where the two differ nothing is allocated.
struct Fields<'a> {
    data: &'a [u8],
    value: u32,
    range: u32,
    shifts: u32,
}

impl<'a> Fields<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut fields = Self {
            data,
            value: 0,
            range: 255,
            shifts: 0,
        };
        fields.value = (fields.byte() << 8) | fields.byte();
        fields
    }

    fn byte(&mut self) -> u32 {
        match self.data.split_first() {
            Some((&byte, rest)) => {
                self.data = rest;
                u32::from(byte)
            }
            None => 0,
        }
    }

    fn flag(&mut self) -> bool {
        let split = 1 + ((self.range - 1) >> 1);
        let bit = self.value >= split << 8;
        if bit {
            self.range -= split;
            self.value -= split << 8;
        } else {
            self.range = split;
        }
        while self.range < 128 {
            self.value <<= 1;
            self.range <<= 1;
            self.shifts += 1;
            if self.shifts == 8 {
                self.shifts = 0;
                self.value |= self.byte();
            }
        }
        bit
    }

    fn literal(&mut self, bits: u32) -> u32 {
        (0..bits).fold(0, |value, _| (value << 1) | u32::from(self.flag()))
    }

    /// An optional signed value: a presence flag, then magnitude and sign.
    fn skip_signed(&mut self, bits: u32) {
        if self.flag() {
            self.literal(bits + 1);
        }
    }
}

/// Checks that a VP8 key frame's partitions fit its chunk: image-webp
/// allocates each declared size before it reads, so a size past the end
/// would claim up to 16 MiB that no byte of the file holds.
fn check_vp8_partitions(data: &[u8]) -> MediaResult<()> {
    let first = (le24(&data[..3]) >> 5) as usize;
    let data = &data[10..];
    if first > data.len() {
        return Err(MediaError::new(format!(
            "WebP VP8 first partition needs {first} bytes, but {} remain",
            data.len()
        )));
    }
    let (header, data) = data.split_at(first);
    let mut fields = Fields::new(header);
    // Color space and clamping type.
    fields.literal(2);
    // Segmentation: whether the map and the feature data update, the
    // feature mode with four quantizer and four filter levels, and the
    // map's three probabilities.
    if fields.flag() {
        let map = fields.flag();
        if fields.flag() {
            fields.flag();
            for bits in [7, 7, 7, 7, 6, 6, 6, 6] {
                fields.skip_signed(bits);
            }
        }
        if map {
            for _ in 0..3 {
                if fields.flag() {
                    fields.literal(8);
                }
            }
        }
    }
    // Filter type, level and sharpness.
    fields.literal(1 + 6 + 3);
    // Loop filter deltas, when enabled and updated: four for reference
    // frames and four for modes.
    if fields.flag() && fields.flag() {
        for _ in 0..8 {
            fields.skip_signed(6);
        }
    }
    let partitions = 1_usize << fields.literal(2);
    let table = 3 * (partitions - 1);
    if data.len() < table {
        return Err(MediaError::new("WebP VP8 partition sizes are truncated"));
    }
    let (sizes, mut rest) = data.split_at(table);
    for (index, size) in sizes.chunks_exact(3).enumerate() {
        let size = le24(size) as usize;
        if size > rest.len() {
            return Err(MediaError::new(format!(
                "WebP VP8 partition {} of {partitions} needs {size} bytes, but {} remain",
                index + 1,
                rest.len()
            )));
        }
        rest = &rest[size..];
    }
    Ok(())
}

/// The size a VP8 key frame or VP8L header declares, once a key frame's
/// partitions are known to fit its chunk.
fn bitstream_size(chunk: Chunk<'_>) -> MediaResult<(u32, u32)> {
    let data = chunk.data;
    match &chunk.fourcc {
        b"VP8 " => {
            if data.len() < 10 || data[3..6] != [0x9d, 0x01, 0x2a] || data[0] & 1 != 0 {
                return Err(MediaError::new("WebP VP8 key frame header is invalid"));
            }
            check_vp8_partitions(data)?;
            Ok((
                u32::from(u16::from_le_bytes([data[6], data[7]]) & 0x3fff),
                u32::from(u16::from_le_bytes([data[8], data[9]]) & 0x3fff),
            ))
        }
        b"VP8L" => {
            if data.len() < 5 || data[0] != 0x2f {
                return Err(MediaError::new("WebP VP8L header is invalid"));
            }
            let bits = u32::from_le_bytes(data[1..5].try_into().expect("four bytes"));
            Ok(((bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1))
        }
        _ => Err(MediaError::new("WebP has no VP8 or VP8L bitstream")),
    }
}

/// What the guarded decode decodes: the canvas, the frame's rectangle on it
/// (the whole canvas for a still image), the frame's bitstream and alpha.
struct Plan<'a> {
    canvas: (u32, u32),
    frame: (u32, u32, u32, u32),
    bitstream: Chunk<'a>,
    alpha: Option<&'a [u8]>,
    /// The VP8X alpha flag: the canvas around an animated frame is
    /// transparent rather than opaque black.
    canvas_alpha: bool,
    exif: bool,
}

impl Plan<'_> {
    /// Bytes of a lossy (VP8) bitstream.
    fn lossy_bytes(&self) -> u64 {
        if &self.bitstream.fourcc == b"VP8 " {
            self.bitstream.data.len() as u64
        } else {
            0
        }
    }

    /// Bytes of lossless-coded data: a VP8L bitstream and an ALPH chunk.
    fn lossless_bytes(&self) -> u64 {
        let bitstream = if &self.bitstream.fourcc == b"VP8L" {
            self.bitstream.data.len()
        } else {
            0
        };
        (bitstream + self.alpha.map_or(0, <[u8]>::len)) as u64
    }
}

fn plan(bytes: &[u8], limits: MediaLimits) -> MediaResult<Plan<'_>> {
    let chunks = riff_chunks(bytes)?;
    let first = *chunks
        .first()
        .ok_or_else(|| MediaError::new("WebP has no chunks"))?;
    if &first.fourcc != b"VP8X" {
        let (width, height) = bitstream_size(first)?;
        limits.check_named_image("WebP", width, height)?;
        return Ok(Plan {
            canvas: (width, height),
            frame: (0, 0, width, height),
            bitstream: first,
            alpha: None,
            canvas_alpha: false,
            exif: false,
        });
    }
    let header = first.data;
    if header.len() < 10 {
        return Err(MediaError::new("WebP VP8X header is truncated"));
    }
    if header[0] & !0x3e != 0 {
        return Err(MediaError::new("WebP VP8X header sets reserved flags"));
    }
    let canvas = (le24(&header[4..7]) + 1, le24(&header[7..10]) + 1);
    limits.check_named_image("WebP", canvas.0, canvas.1)?;
    let (frame, inner) = if header[0] & 0x02 != 0 {
        let anmf = chunks
            .iter()
            .find(|chunk| &chunk.fourcc == b"ANMF")
            .ok_or_else(|| MediaError::new("animated WebP has no frame"))?;
        let data = anmf.data;
        if data.len() < 16 {
            return Err(MediaError::new("WebP ANMF header is truncated"));
        }
        let frame = (
            le24(&data[0..3]) * 2,
            le24(&data[3..6]) * 2,
            le24(&data[6..9]) + 1,
            le24(&data[9..12]) + 1,
        );
        if u64::from(frame.0) + u64::from(frame.2) > u64::from(canvas.0)
            || u64::from(frame.1) + u64::from(frame.3) > u64::from(canvas.1)
        {
            return Err(MediaError::new(format!(
                "WebP frame {} x {} at ({}, {}) exceeds its {} x {} canvas",
                frame.2, frame.3, frame.0, frame.1, canvas.0, canvas.1
            )));
        }
        (frame, sub_chunks(&data[16..])?)
    } else {
        ((0, 0, canvas.0, canvas.1), chunks[1..].to_vec())
    };
    let bitstream = *inner
        .iter()
        .find(|chunk| &chunk.fourcc == b"VP8 " || &chunk.fourcc == b"VP8L")
        .ok_or_else(|| MediaError::new("WebP has no VP8 or VP8L bitstream"))?;
    let size = bitstream_size(bitstream)?;
    if size != (frame.2, frame.3) {
        return Err(MediaError::new(format!(
            "WebP bitstream {} x {} differs from its {} x {} frame",
            size.0, size.1, frame.2, frame.3
        )));
    }
    let alpha = if &bitstream.fourcc == b"VP8 " {
        inner
            .iter()
            .find(|chunk| &chunk.fourcc == b"ALPH")
            .map(|chunk| chunk.data)
    } else {
        None
    };
    Ok(Plan {
        canvas,
        frame,
        bitstream,
        alpha,
        canvas_alpha: header[0] & 0x10 != 0,
        exif: header[0] & 0x08 != 0 || chunks.iter().any(|chunk| &chunk.fourcc == b"EXIF"),
    })
}

fn checked_plan(
    bytes: &[u8],
    limits: MediaLimits,
    orientation: Orientation,
) -> MediaResult<Plan<'_>> {
    limits.check_bytes(bytes.len())?;
    let plan = plan(bytes, limits)?;
    if orientation == Orientation::Exif && plan.exif {
        return Err(MediaError::new("WebP EXIF orientation is not supported"));
    }
    Ok(plan)
}

/// The canvas size from the headers, checked against `limits`.
pub fn probe(
    bytes: &[u8],
    limits: MediaLimits,
    orientation: Orientation,
) -> MediaResult<(u32, u32)> {
    Ok(checked_plan(bytes, limits, orientation)?.canvas)
}

/// Decoder storage derived from the headers, before any allocation: the
/// RGB canvas, the rebuilt still file and image-webp's reads of it (each
/// at most the source), [`FRAME_BYTES_PER_PIXEL`] over the
/// macroblock-padded frame, [`LOSSY_BYTES_PER_BYTE`] and
/// [`LOSSLESS_BYTES_PER_BYTE`] over the coded data, and a fixed allowance
/// for the decoders' state.
pub fn decode_storage_bytes(
    bytes: &[u8],
    limits: MediaLimits,
    orientation: Orientation,
) -> MediaResult<u64> {
    let plan = checked_plan(bytes, limits, orientation)?;
    let padded = |size: u32| u64::from(size.div_ceil(16) * 16);
    Ok(u64::from(plan.canvas.0) * u64::from(plan.canvas.1) * 3
        + 2 * bytes.len() as u64
        + padded(plan.frame.2) * padded(plan.frame.3) * FRAME_BYTES_PER_PIXEL
        + plan.lossy_bytes() * LOSSY_BYTES_PER_BYTE
        + plan.lossless_bytes() * LOSSLESS_BYTES_PER_BYTE
        + 256 * 1024)
}

/// A standalone still WebP of the planned frame: its bitstream, with its
/// alpha under a VP8X header when it has one. It is never larger than the
/// source.
fn still(plan: &Plan<'_>) -> MediaResult<Vec<u8>> {
    let padded = |data: &[u8]| 8 + data.len() + (data.len() & 1);
    let mut vp8x = [0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    vp8x[4..7].copy_from_slice(&(plan.frame.2 - 1).to_le_bytes()[..3]);
    vp8x[7..10].copy_from_slice(&(plan.frame.3 - 1).to_le_bytes()[..3]);
    let mut chunks: Vec<([u8; 4], &[u8])> = Vec::new();
    if let Some(alpha) = plan.alpha {
        chunks.push((*b"VP8X", &vp8x));
        chunks.push((*b"ALPH", alpha));
    }
    chunks.push((plan.bitstream.fourcc, plan.bitstream.data));
    let body = 4 + chunks.iter().map(|(_, data)| padded(data)).sum::<usize>();
    let mut out = try_with_capacity(8 + body)?;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(body as u32).to_le_bytes());
    out.extend_from_slice(b"WEBP");
    for (fourcc, data) in chunks {
        out.extend_from_slice(&fourcc);
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        if data.len() % 2 == 1 {
            out.push(0);
        }
    }
    Ok(out)
}

fn invalid(error: image_webp::DecodingError) -> MediaError {
    MediaError::new(format!("WebP image data is invalid: {error}"))
}

/// Decodes the still image or frame 0 onto its canvas. [`Alpha::Discard`]
/// keeps the color channels, as Pillow's `convert("RGB")` does;
/// [`Alpha::CompositeWhite`] composites the decoded alpha, and a
/// transparent canvas around an animated frame, onto white. An EXIF
/// orientation tag is honored only by refusal: under
/// [`Orientation::Exif`] a file with EXIF metadata is refused.
pub fn decode(
    bytes: &[u8],
    limits: MediaLimits,
    alpha: Alpha,
    orientation: Orientation,
) -> MediaResult<RgbImage> {
    let plan = checked_plan(bytes, limits, orientation)?;
    let still = still(&plan)?;
    let mut decoder =
        image_webp::WebPDecoder::new(Cursor::new(still.as_slice())).map_err(invalid)?;
    let (left, top, frame_width, frame_height) = plan.frame;
    if decoder.dimensions() != (frame_width, frame_height) {
        return Err(MediaError::new("WebP frame size differs from its header"));
    }
    let channels = if decoder.has_alpha() { 4 } else { 3 };
    let frame_pixels = frame_width as usize * frame_height as usize;
    if decoder.output_buffer_size() != Some(frame_pixels * channels) {
        return Err(MediaError::new("WebP frame size differs from its header"));
    }
    let mut decoded = try_filled(frame_pixels * channels, 0_u8)?;
    decoder.read_image(&mut decoded).map_err(invalid)?;
    let (width, height) = plan.canvas;
    let background = if alpha == Alpha::CompositeWhite && plan.canvas_alpha {
        255
    } else {
        0
    };
    let mut rgb = try_filled(width as usize * height as usize * 3, background)?;
    for (y, row) in decoded
        .chunks_exact(frame_width as usize * channels)
        .enumerate()
    {
        let start = ((top as usize + y) * width as usize + left as usize) * 3;
        for (target, source) in rgb[start..start + frame_width as usize * 3]
            .chunks_exact_mut(3)
            .zip(row.chunks_exact(channels))
        {
            let color = [source[0], source[1], source[2]];
            target.copy_from_slice(&match channels {
                4 => composite(color, source[3], alpha),
                _ => color,
            });
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

    fn lossless(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        image_webp::WebPEncoder::new(&mut out)
            .encode(rgba, width, height, image_webp::ColorType::Rgba8)
            .unwrap();
        out
    }

    fn riff(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let mut body = b"WEBP".to_vec();
        for (fourcc, data) in chunks {
            body.extend_from_slice(*fourcc);
            body.extend_from_slice(&(data.len() as u32).to_le_bytes());
            body.extend_from_slice(data);
            if data.len() % 2 == 1 {
                body.push(0);
            }
        }
        [
            b"RIFF".as_slice(),
            &(body.len() as u32).to_le_bytes(),
            &body,
        ]
        .concat()
    }

    fn vp8x(flags: u8, width: u32, height: u32) -> Vec<u8> {
        [
            [flags, 0, 0, 0].as_slice(),
            &(width - 1).to_le_bytes()[..3],
            &(height - 1).to_le_bytes()[..3],
        ]
        .concat()
    }

    /// An animated file whose first frame is `bitstream` (a VP8L chunk's
    /// data) at (`x`, `y`) on a `width` x `height` canvas.
    fn animated(
        width: u32,
        height: u32,
        x: u32,
        y: u32,
        size: (u32, u32),
        bitstream: &[u8],
    ) -> Vec<u8> {
        let mut anmf = Vec::new();
        for value in [x / 2, y / 2, size.0 - 1, size.1 - 1, 0] {
            anmf.extend_from_slice(&value.to_le_bytes()[..3]);
        }
        // Blending and disposal flags.
        anmf.push(0);
        anmf.extend_from_slice(b"VP8L");
        anmf.extend_from_slice(&(bitstream.len() as u32).to_le_bytes());
        anmf.extend_from_slice(bitstream);
        if bitstream.len() % 2 == 1 {
            anmf.push(0);
        }
        riff(&[
            (b"VP8X", &vp8x(0x12, width, height)),
            (b"ANIM", &[0; 6]),
            (b"ANMF", &anmf),
        ])
    }

    fn vp8l(file: &[u8]) -> &[u8] {
        let chunks = riff_chunks(file).unwrap();
        assert_eq!(&chunks[0].fourcc, b"VP8L");
        chunks[0].data
    }

    #[test]
    fn still_and_first_frame_decode_as_pillow_composes_them() {
        let rgba = [10, 20, 30, 255, 40, 50, 60, 0];
        let still = lossless(2, 1, &rgba);
        let stored = Orientation::Stored;
        assert_eq!(probe(&still, limits(), Orientation::Exif).unwrap(), (2, 1));
        let image = decode(&still, limits(), Alpha::Discard, stored).unwrap();
        assert_eq!(image.data, [10, 20, 30, 40, 50, 60]);
        let white = Alpha::CompositeWhite;
        assert_eq!(
            decode(&still, limits(), white, stored).unwrap().data,
            [10, 20, 30, 255, 255, 255]
        );
        // Frame 0 at (2, 0) on a 4 x 2 canvas; the rest of the canvas is
        // transparent black.
        let file = animated(4, 2, 2, 0, (2, 1), vp8l(&still));
        let image = decode(&file, limits(), Alpha::Discard, stored).unwrap();
        let mut expected = vec![0; 4 * 2 * 3];
        expected[6..12].copy_from_slice(&[10, 20, 30, 40, 50, 60]);
        assert_eq!((image.width, image.height, image.data), (4, 2, expected));
        assert_eq!(
            decode(&file, limits(), white, stored).unwrap().data[..6],
            [255; 6]
        );
    }

    #[test]
    fn container_conflicts_and_reserved_flags_are_refused() {
        let still = lossless(2, 1, &[0; 8]);
        let bitstream = vp8l(&still);
        // A frame past its canvas, or one whose bitstream names another size.
        for (file, message) in [
            (
                animated(2, 2, 2, 0, (2, 1), bitstream),
                "WebP frame 2 x 1 at (2, 0) exceeds its 2 x 2 canvas",
            ),
            (
                animated(4, 4, 0, 0, (3, 1), bitstream),
                "WebP bitstream 2 x 1 differs from its 3 x 1 frame",
            ),
            (
                riff(&[(b"VP8X", &vp8x(0, 4, 4)), (b"VP8L", bitstream)]),
                "WebP bitstream 2 x 1 differs from its 4 x 4 frame",
            ),
            (
                riff(&[(b"VP8X", &vp8x(0x01, 2, 1)), (b"VP8L", bitstream)]),
                "WebP VP8X header sets reserved flags",
            ),
            (
                riff(&[(b"VP8X", &vp8x(0, 16384, 16384)), (b"VP8L", bitstream)]),
                "WebP image 16384 x 16384 has 268435456 pixels, above the 1048576-pixel limit",
            ),
        ] {
            assert_eq!(
                decode(&file, limits(), Alpha::Discard, Orientation::Stored)
                    .unwrap_err()
                    .to_string(),
                message
            );
        }
        let exif = riff(&[
            (b"VP8X", &vp8x(0x08, 2, 1)),
            (b"VP8L", bitstream),
            (b"EXIF", b"MM\0*"),
        ]);
        assert!(probe(&exif, limits(), Orientation::Stored).is_ok());
        assert_eq!(
            probe(&exif, limits(), Orientation::Exif)
                .unwrap_err()
                .to_string(),
            "WebP EXIF orientation is not supported"
        );
    }

    /// RFC 6386's boolean encoder at even probability.
    fn bool_coded(bits: &[bool]) -> Vec<u8> {
        fn carry(out: &mut [u8]) {
            for byte in out.iter_mut().rev() {
                let (sum, overflow) = byte.overflowing_add(1);
                *byte = sum;
                if !overflow {
                    break;
                }
            }
        }
        let mut out = Vec::new();
        let (mut range, mut bottom, mut count) = (255_u32, 0_u32, 24_u32);
        for &bit in bits {
            let split = 1 + ((range - 1) >> 1);
            if bit {
                bottom = bottom.wrapping_add(split);
                range -= split;
            } else {
                range = split;
            }
            while range < 128 {
                range <<= 1;
                if bottom & (1 << 31) != 0 {
                    carry(&mut out);
                }
                bottom <<= 1;
                count -= 1;
                if count == 0 {
                    out.push((bottom >> 24) as u8);
                    bottom &= (1 << 24) - 1;
                    count = 8;
                }
            }
        }
        if bottom & (1 << (32 - count)) != 0 {
            carry(&mut out);
        }
        let mut value = bottom << (count & 7);
        for _ in 0..count >> 3 {
            value <<= 8;
        }
        for _ in 0..4 {
            out.push((value >> 24) as u8);
            value <<= 8;
        }
        out
    }

    /// A 16 x 16 VP8 key frame whose first partition codes `fields`, each
    /// a value and its bit count, followed by `rest`.
    fn vp8(fields: &[(u32, u32)], rest: &[u8]) -> Vec<u8> {
        let bits = fields
            .iter()
            .flat_map(|&(value, count)| (0..count).rev().map(move |bit| (value >> bit) & 1 == 1))
            .collect::<Vec<_>>();
        let header = bool_coded(&bits);
        // A shown key frame, then the start code and size.
        let tag = ((header.len() as u32) << 5) | 0x10;
        let mut data = tag.to_le_bytes()[..3].to_vec();
        data.extend_from_slice(&[0x9d, 0x01, 0x2a, 16, 0, 16, 0]);
        data.extend_from_slice(&header);
        data.extend_from_slice(rest);
        data
    }

    fn probe_vp8(data: &[u8]) -> MediaResult<(u32, u32)> {
        probe(&riff(&[(b"VP8 ", data)]), limits(), Orientation::Stored)
    }

    #[test]
    fn vp8_header_fields_decode_as_coded() {
        let mut state = 0x2545_f491_u32;
        let bits = (0..4000)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state % 7 < 2
            })
            .collect::<Vec<_>>();
        let coded = bool_coded(&bits);
        let mut fields = Fields::new(&coded);
        for (index, &bit) in bits.iter().enumerate() {
            assert_eq!(fields.flag(), bit, "{index}");
        }
    }

    #[test]
    fn vp8_partitions_past_their_chunk_are_refused() {
        // Four partitions; no segmentation or loop filter deltas.
        let four = [(0, 2), (0, 1), (0, 10), (0, 1), (2, 2)];
        let sizes = [1, 0, 0, 1, 0, 0, 0xe8, 0x03, 0];
        let message = |data: &[u8]| probe_vp8(data).unwrap_err().to_string();
        assert_eq!(
            message(&vp8(&four, &[sizes.as_slice(), &[0; 10]].concat())),
            "WebP VP8 partition 3 of 4 needs 1000 bytes, but 8 remain"
        );
        assert_eq!(
            message(&vp8(&four, &[0; 4])),
            "WebP VP8 partition sizes are truncated"
        );
        let mut data = vp8(&four, &[0; 6]);
        data[..3].copy_from_slice(&((5000_u32 << 5) | 0x10).to_le_bytes()[..3]);
        let remain = data.len() - 10;
        assert_eq!(
            message(&data),
            format!("WebP VP8 first partition needs 5000 bytes, but {remain} remain")
        );
        // Eight partitions after every optional field: segment quantizer
        // and filter levels, map probabilities and loop filter deltas.
        let mut eight = vec![(0, 2), (1, 1), (1, 1), (1, 1), (1, 1)];
        for level in [(0x55, 7), (0x2a, 7), (0, 7), (0x7f, 7)] {
            eight.extend([(1, 1), level, (1, 1)]);
        }
        eight.extend([(0, 1), (1, 1), (33, 6), (0, 1), (0, 1), (0, 1)]);
        eight.extend([(1, 1), (200, 8), (0, 1), (1, 1), (7, 8)]);
        eight.extend([(0, 10), (1, 1), (1, 1)]);
        for delta in 0..8 {
            eight.extend([(1, 1), (delta * 7, 6), (delta & 1, 1)]);
        }
        eight.push((3, 2));
        let mut sizes = [0; 21];
        assert_eq!(probe_vp8(&vp8(&eight, &sizes)).unwrap(), (16, 16));
        sizes[18] = 1;
        assert_eq!(
            message(&vp8(&eight, &sizes)),
            "WebP VP8 partition 7 of 8 needs 1 bytes, but 0 remain"
        );
    }

    #[test]
    fn truncations_and_mutations_never_panic() {
        let still = lossless(7, 5, &(0..140).map(|v| (v * 37) as u8).collect::<Vec<_>>());
        let decode =
            |bytes: &[u8]| decode(bytes, limits(), Alpha::CompositeWhite, Orientation::Stored);
        assert!(decode(&still).is_ok());
        for cut in 0..still.len() {
            assert!(decode(&still[..cut]).is_err(), "{cut}");
        }
        let mut state = 0x9e37_79b9_u32;
        for _ in 0..2000 {
            let mut bytes = still.clone();
            for _ in 0..3 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let index = state as usize % bytes.len();
                bytes[index] ^= (state >> 24) as u8 | 1;
            }
            let _ = decode(&bytes);
        }
    }
}
