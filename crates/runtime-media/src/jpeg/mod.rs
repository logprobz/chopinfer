//! Bounded 8-bit JPEG decoding. The scan decoder and reconstruction live beside
//! this marker parser so probing and full decoding agree on frame geometry.

use crate::{MediaError, MediaLimits, MediaResult, Orientation, RgbImage};

mod decode;
mod entropy;
mod idct;
mod render;

const MAX_JPEG_PIXELS: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Header {
    pub width: u16,
    pub height: u16,
    pub components: u8,
    pub orientation: u8,
    pub progressive: bool,
}

impl Header {
    fn oriented_size(self) -> (u32, u32) {
        if matches!(self.orientation, 5..=8) {
            (u32::from(self.height), u32::from(self.width))
        } else {
            (u32::from(self.width), u32::from(self.height))
        }
    }
}

fn be16(bytes: &[u8]) -> u16 {
    u16::from_be_bytes([bytes[0], bytes[1]])
}

pub(crate) struct Markers<'a> {
    bytes: &'a [u8],
    pub pos: usize,
}

impl<'a> Markers<'a> {
    pub fn new(bytes: &'a [u8]) -> MediaResult<Self> {
        if !bytes.starts_with(&[0xff, 0xd8]) {
            return Err(MediaError::new("JPEG SOI is missing"));
        }
        Ok(Self { bytes, pos: 2 })
    }

    pub fn next(&mut self) -> MediaResult<(u8, &'a [u8])> {
        if self.bytes.get(self.pos) != Some(&0xff) {
            return Err(MediaError::new("invalid JPEG marker"));
        }
        while self.bytes.get(self.pos) == Some(&0xff) {
            self.pos += 1;
        }
        let marker = *self
            .bytes
            .get(self.pos)
            .ok_or_else(|| MediaError::new("truncated JPEG marker"))?;
        self.pos += 1;
        if marker == 0x00 || marker == 0xd8 || matches!(marker, 0xd0..=0xd7) {
            return Err(MediaError::new("unexpected JPEG marker"));
        }
        if marker == 0xd9 || marker == 0x01 {
            return Ok((marker, &[]));
        }
        let length = self
            .bytes
            .get(self.pos..self.pos + 2)
            .map(be16)
            .ok_or_else(|| MediaError::new("truncated JPEG segment length"))?
            as usize;
        if length < 2 {
            return Err(MediaError::new("invalid JPEG segment length"));
        }
        let end = self
            .pos
            .checked_add(length)
            .ok_or_else(|| MediaError::new("JPEG segment overflow"))?;
        let payload = self
            .bytes
            .get(self.pos + 2..end)
            .ok_or_else(|| MediaError::new("truncated JPEG segment"))?;
        self.pos = end;
        Ok((marker, payload))
    }
}

pub(crate) fn frame(marker: u8, payload: &[u8], orientation: u8) -> MediaResult<Header> {
    if !matches!(marker, 0xc0 | 0xc2) {
        return Err(MediaError::new("unsupported JPEG frame coding"));
    }
    if payload.len() < 6 || payload[0] != 8 {
        return Err(MediaError::new("unsupported JPEG precision or SOF"));
    }
    let height = be16(&payload[1..3]);
    let width = be16(&payload[3..5]);
    let components = payload[5];
    if width == 0
        || height == 0
        || !matches!(components, 1 | 3 | 4)
        || payload.len() != 6 + usize::from(components) * 3
    {
        return Err(MediaError::new("invalid JPEG frame geometry"));
    }
    let mut seen = [false; 256];
    let mut sampling_units = 0_u32;
    for entry in payload[6..].chunks_exact(3) {
        let h = entry[1] >> 4;
        let v = entry[1] & 15;
        if seen[entry[0] as usize] || !(1..=4).contains(&h) || !(1..=4).contains(&v) || entry[2] > 3
        {
            return Err(MediaError::new("invalid JPEG component descriptor"));
        }
        seen[entry[0] as usize] = true;
        sampling_units += u32::from(h) * u32::from(v);
    }
    if sampling_units > 10 {
        return Err(MediaError::new("JPEG sampling exceeds MCU limit"));
    }
    Ok(Header {
        width,
        height,
        components,
        orientation,
        progressive: marker == 0xc2,
    })
}

fn tiff_orientation(bytes: &[u8]) -> Option<u8> {
    let tiff = bytes.strip_prefix(b"Exif\0\0")?;
    let little = match tiff.get(..2)? {
        b"II" => true,
        b"MM" => false,
        _ => return None,
    };
    let read16 = |offset: usize| -> Option<u16> {
        let bytes: [u8; 2] = tiff.get(offset..offset.checked_add(2)?)?.try_into().ok()?;
        Some(if little {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        })
    };
    let read32 = |offset: usize| -> Option<u32> {
        let bytes: [u8; 4] = tiff.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
        Some(if little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        })
    };
    if read16(2)? != 42 {
        return None;
    }
    let ifd = usize::try_from(read32(4)?).ok()?;
    let count = usize::from(read16(ifd)?);
    let entries = ifd.checked_add(2)?;
    if count > (tiff.len().checked_sub(entries)?) / 12 {
        return None;
    }
    for index in 0..count {
        let offset = entries + index * 12;
        if read16(offset)? == 0x0112 && read16(offset + 2)? == 3 && read32(offset + 4)? == 1 {
            let orientation = u8::try_from(read16(offset + 8)?).ok()?;
            return (1..=8).contains(&orientation).then_some(orientation);
        }
    }
    None
}

pub fn decode(bytes: &[u8], limits: MediaLimits) -> MediaResult<RgbImage> {
    decode_oriented(bytes, limits, Orientation::Exif)
}

/// [`decode`] under `orientation`: [`Orientation::Stored`] ignores an EXIF
/// orientation tag and keeps the stored pixels.
pub fn decode_oriented(
    bytes: &[u8],
    limits: MediaLimits,
    orientation: Orientation,
) -> MediaResult<RgbImage> {
    let mut frame = decode::coefficients(bytes, limits)?;
    if orientation == Orientation::Stored {
        frame.header.orientation = 1;
    }
    render::render(frame, limits)
}

/// Channel-level comparison against a captured libjpeg-turbo RGB frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CaptureParity {
    pub mismatched_channels: usize,
    pub max_abs_difference: u8,
    pub first_mismatch: Option<usize>,
}

pub fn compare_capture(
    bytes: &[u8],
    expected: &RgbImage,
    limits: MediaLimits,
) -> MediaResult<CaptureParity> {
    let decoded = decode(bytes, limits)?;
    if (decoded.width, decoded.height) != (expected.width, expected.height)
        || decoded.data.len() != expected.data.len()
    {
        return Err(MediaError::new("JPEG capture geometry differs"));
    }
    let mut parity = CaptureParity {
        mismatched_channels: 0,
        max_abs_difference: 0,
        first_mismatch: None,
    };
    for (index, (&actual, &reference)) in decoded.data.iter().zip(&expected.data).enumerate() {
        let difference = actual.abs_diff(reference);
        if difference != 0 {
            parity.mismatched_channels += 1;
            parity.max_abs_difference = parity.max_abs_difference.max(difference);
            parity.first_mismatch.get_or_insert(index);
        }
    }
    Ok(parity)
}

pub fn probe(bytes: &[u8], limits: MediaLimits) -> MediaResult<(u32, u32)> {
    probe_oriented(bytes, limits, Orientation::Exif)
}

/// [`probe`] under `policy`: [`Orientation::Stored`] gives the stored
/// width and height whatever the EXIF tag says.
pub fn probe_oriented(
    bytes: &[u8],
    limits: MediaLimits,
    policy: Orientation,
) -> MediaResult<(u32, u32)> {
    limits.check_bytes(bytes.len())?;
    let mut markers = Markers::new(bytes)?;
    let mut orientation = 1;
    let mut header: Option<Header> = None;
    loop {
        let (marker, payload) = markers.next()?;
        match marker {
            0xe1 => {
                if let Some(value) = tiff_orientation(payload) {
                    orientation = value;
                    if let Some(current) = &mut header {
                        current.orientation = value;
                    }
                }
            }
            0xc0 | 0xc2 => {
                if header.is_some() {
                    return Err(MediaError::new("duplicate JPEG frame"));
                }
                let parsed = frame(marker, payload, orientation)?;
                limits.check_image(u32::from(parsed.width), u32::from(parsed.height))?;
                if u64::from(parsed.width) * u64::from(parsed.height) > MAX_JPEG_PIXELS {
                    return Err(MediaError::new("JPEG decoded pixel hard limit exceeded"));
                }
                header = Some(parsed);
            }
            0xc1 | 0xc3..=0xcf if marker != 0xc4 && marker != 0xc8 && marker != 0xcc => {
                return Err(MediaError::new("unsupported JPEG frame coding"));
            }
            0xda => {
                let mut header = header.ok_or_else(|| MediaError::new("JPEG SOF is missing"))?;
                if policy == Orientation::Stored {
                    header.orientation = 1;
                }
                return Ok(header.oriented_size());
            }
            0xd9 => return Err(MediaError::new("JPEG scan is missing")),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> MediaLimits {
        MediaLimits {
            maximum_bytes: 1024,
            maximum_pixels: 1024,
            maximum_audio_frames: 0,
            maximum_frames: 0,
            maximum_tokens: 0,
            maximum_aspect_ratio: 200.0,
        }
    }

    fn header() -> Vec<u8> {
        vec![
            0xff, 0xd8, 0xff, 0xc0, 0, 11, 8, 0, 2, 0, 3, 1, 1, 0x11, 0, 0xff, 0xda, 0, 2,
        ]
    }

    fn segment(bytes: &mut Vec<u8>, marker: u8, payload: &[u8]) {
        bytes.extend_from_slice(&[0xff, marker]);
        bytes.extend_from_slice(&u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
        bytes.extend_from_slice(payload);
    }

    fn simple_image(progressive: bool) -> Vec<u8> {
        let mut bytes = vec![0xff, 0xd8];
        let mut quant = vec![0];
        quant.extend_from_slice(&[1; 64]);
        segment(&mut bytes, 0xdb, &quant);
        segment(
            &mut bytes,
            if progressive { 0xc2 } else { 0xc0 },
            &[8, 0, 8, 0, 8, 1, 1, 0x11, 0],
        );
        let mut huffman = vec![0, 1];
        huffman.extend_from_slice(&[0; 15]);
        huffman.push(0);
        huffman.extend_from_slice(&[0x10, 1]);
        huffman.extend_from_slice(&[0; 15]);
        huffman.push(0);
        segment(&mut bytes, 0xc4, &huffman);
        if progressive {
            segment(&mut bytes, 0xda, &[1, 1, 0, 0, 0, 0]);
            bytes.push(0x7f);
            segment(&mut bytes, 0xda, &[1, 1, 0, 1, 63, 0]);
            bytes.push(0x7f);
        } else {
            segment(&mut bytes, 0xda, &[1, 1, 0, 0, 63, 0]);
            bytes.push(0x3f);
        }
        bytes.extend_from_slice(&[0xff, 0xd9]);
        bytes
    }

    fn entropy_bits(bits: &str) -> Vec<u8> {
        let mut encoded = Vec::new();
        let mut byte = 0_u8;
        let mut count = 0;
        for bit in bits.bytes() {
            byte = (byte << 1) | u8::from(bit == b'1');
            count += 1;
            if count == 8 {
                encoded.push(byte);
                if byte == 0xff {
                    encoded.push(0);
                }
                byte = 0;
                count = 0;
            }
        }
        if count != 0 {
            let padding = 8 - count;
            let padded = (byte << padding) | ((1 << padding) - 1);
            encoded.push(padded);
            if padded == 0xff {
                encoded.push(0);
            }
        }
        encoded
    }

    #[test]
    fn baseline_and_progressive_zero_blocks_decode_to_gray() {
        for progressive in [false, true] {
            let bytes = simple_image(progressive);
            let image = decode(&bytes, limits()).unwrap();
            assert_eq!((image.width, image.height), (8, 8));
            assert_eq!(image.data, vec![128; 8 * 8 * 3]);
        }
    }

    #[test]
    fn capture_hook_counts_exact_channel_differences() {
        let bytes = simple_image(false);
        let mut expected = decode(&bytes, limits()).unwrap();
        assert_eq!(
            compare_capture(&bytes, &expected, limits())
                .unwrap()
                .mismatched_channels,
            0
        );
        expected.data[3] += 1;
        assert_eq!(
            compare_capture(&bytes, &expected, limits()).unwrap(),
            CaptureParity {
                mismatched_channels: 1,
                max_abs_difference: 1,
                first_mismatch: Some(3)
            }
        );
    }

    #[test]
    fn hard_pixel_limit_rejects_oversized_sof_before_allocation() {
        let mut bytes = simple_image(false);
        let sof = bytes
            .windows(2)
            .position(|pair| pair == [0xff, 0xc0])
            .unwrap();
        bytes[sof + 5..sof + 9].copy_from_slice(&[0xff, 0xff, 0xff, 0xff]);
        let mut permissive = limits();
        permissive.maximum_pixels = u64::MAX;
        assert!(
            probe(&bytes, permissive)
                .unwrap_err()
                .to_string()
                .contains("hard limit")
        );
        assert!(
            decode(&bytes, permissive)
                .unwrap_err()
                .to_string()
                .contains("hard limit")
        );
    }

    #[test]
    fn progressive_ac_refinement_changes_islow_samples() {
        let mut bytes = vec![0xff, 0xd8];
        let mut quant = vec![0];
        quant.extend_from_slice(&[40; 64]);
        segment(&mut bytes, 0xdb, &quant);
        segment(&mut bytes, 0xc2, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0]);
        let mut huffman = vec![0, 1];
        huffman.extend_from_slice(&[0; 15]);
        huffman.push(0);
        huffman.extend_from_slice(&[0x10, 2]);
        huffman.extend_from_slice(&[0; 15]);
        huffman.extend_from_slice(&[0x01, 0x00]);
        segment(&mut bytes, 0xc4, &huffman);
        segment(&mut bytes, 0xda, &[1, 1, 0, 0, 0, 0]);
        bytes.extend_from_slice(&entropy_bits("0"));
        segment(&mut bytes, 0xda, &[1, 1, 0, 1, 63, 1]);
        bytes.extend_from_slice(&entropy_bits("011"));
        segment(&mut bytes, 0xda, &[1, 1, 0, 1, 63, 0x10]);
        bytes.extend_from_slice(&entropy_bits("11"));
        bytes.extend_from_slice(&[0xff, 0xd9]);
        let image = decode(&bytes, limits()).unwrap();
        assert_ne!(image.data[0], image.data[7 * 3]);
    }

    #[test]
    fn restart_interval_resets_prediction_and_exif_rotates_pixels() {
        let mut bytes = vec![0xff, 0xd8];
        let mut exif =
            b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0".to_vec();
        segment(&mut bytes, 0xe1, &exif);
        exif.clear();
        let mut quant = vec![0];
        quant.extend_from_slice(&[1; 64]);
        segment(&mut bytes, 0xdb, &quant);
        segment(&mut bytes, 0xc0, &[8, 0, 16, 0, 8, 1, 1, 0x11, 0]);
        let mut huffman = vec![0, 1];
        huffman.extend_from_slice(&[0; 15]);
        huffman.push(7);
        huffman.extend_from_slice(&[0x10, 1]);
        huffman.extend_from_slice(&[0; 15]);
        huffman.push(0);
        segment(&mut bytes, 0xc4, &huffman);
        segment(&mut bytes, 0xda, &[1, 1, 0, 0, 63, 0]);
        bytes.extend_from_slice(&entropy_bits("010000000001111110"));
        bytes.extend_from_slice(&[0xff, 0xd9]);
        let image = decode(&bytes, limits()).unwrap();
        assert_eq!((image.width, image.height), (16, 8));
        assert_eq!(image.data[0], 128);
        assert_eq!(image.data[15 * 3], 136);

        let mut restarted = simple_image(false);
        let sof = restarted
            .windows(2)
            .position(|pair| pair == [0xff, 0xc0])
            .unwrap();
        restarted[sof + 8] = 16;
        let sos = restarted
            .windows(2)
            .position(|pair| pair == [0xff, 0xda])
            .unwrap();
        restarted.splice(sos..sos, [0xff, 0xdd, 0, 4, 0, 1]);
        let eoi = restarted.len() - 2;
        restarted.splice(eoi..eoi, [0xff, 0xd0, 0x3f]);
        assert_eq!(
            decode(&restarted, limits()).unwrap().data,
            vec![128; 16 * 8 * 3]
        );
    }

    #[test]
    fn truncated_and_mutated_jpeg_never_panics() {
        for progressive in [false, true] {
            let bytes = simple_image(progressive);
            for end in 0..bytes.len() {
                assert!(
                    decode(&bytes[..end], limits()).is_err(),
                    "truncated at {end}"
                );
            }
            let mut state = 0x1234_5678_u32;
            for _ in 0..2000 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let mut corrupt = bytes.clone();
                let position = (state as usize) % corrupt.len();
                corrupt[position] ^= (state >> 16) as u8;
                assert!(std::panic::catch_unwind(|| decode(&corrupt, limits())).is_ok());
            }
            for length in 0..128 {
                let mut random = vec![0xff, 0xd8];
                for _ in 0..length {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    random.push((state >> 24) as u8);
                }
                assert!(std::panic::catch_unwind(|| decode(&random, limits())).is_ok());
            }
        }
    }

    #[test]
    fn probe_enforces_geometry_and_segment_bounds() {
        let bytes = header();
        assert_eq!(probe(&bytes, limits()).unwrap(), (3, 2));
        for end in 0..bytes.len() {
            assert!(
                probe(&bytes[..end], limits()).is_err(),
                "truncated at {end}"
            );
        }
        let mut oversized = bytes.clone();
        oversized[9] = 10;
        assert!(probe(&oversized, limits()).is_err());
    }

    #[test]
    fn exif_orientation_swaps_probe_geometry() {
        let mut bytes = vec![0xff, 0xd8, 0xff, 0xe1, 0, 34];
        bytes.extend_from_slice(
            b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0",
        );
        bytes.extend_from_slice(&header()[2..]);
        assert_eq!(probe(&bytes, limits()).unwrap(), (2, 3));
        // Kept as stored, the tag changes nothing.
        assert_eq!(
            probe_oriented(&bytes, limits(), Orientation::Stored).unwrap(),
            (3, 2)
        );
    }
}
