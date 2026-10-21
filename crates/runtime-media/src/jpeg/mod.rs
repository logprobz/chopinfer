//! Bounded 8-bit JPEG decoding. The scan decoder and reconstruction live beside
//! this marker parser so probing and full decoding agree on frame geometry.

use crate::{MediaError, MediaLimits, MediaResult, Orientation, RgbImage, orientation};

mod decode;
mod entropy;
mod idct;
mod render;

/// The coefficient storage one JPEG may need: four full-resolution
/// components of the largest image the host admits (89,478,485 pixels) at
/// one `i16` per coefficient, with MCU padding.
const MAX_COEFFICIENT_BYTES: u64 = 768 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Header {
    pub width: u16,
    pub height: u16,
    pub components: u8,
    pub orientation: u8,
    pub progressive: bool,
    pub sampling: [(u8, u8); 4],
}

impl Header {
    fn maximum_sampling(self) -> (usize, usize) {
        let used = &self.sampling[..usize::from(self.components)];
        (
            usize::from(used.iter().map(|s| s.0).max().unwrap_or(1)),
            usize::from(used.iter().map(|s| s.1).max().unwrap_or(1)),
        )
    }

    /// Whole MCU padding used by both baseline and progressive decoding.
    fn component_blocks(self, index: usize) -> (usize, usize) {
        let (max_h, max_v) = self.maximum_sampling();
        let (h, v) = self.sampling[index];
        (
            usize::from(self.width).div_ceil(8 * max_h) * usize::from(h),
            usize::from(self.height).div_ceil(8 * max_v) * usize::from(v),
        )
    }

    /// The padded blocks of every component, refusing fractional sampling.
    fn blocks(self) -> MediaResult<u64> {
        let (max_h, max_v) = self.maximum_sampling();
        let mut blocks = 0_u64;
        for index in 0..usize::from(self.components) {
            let (h, v) = self.sampling[index];
            if max_h % usize::from(h) != 0 || max_v % usize::from(v) != 0 {
                return Err(MediaError::new("unsupported fractional JPEG sampling"));
            }
            let (w, h) = self.component_blocks(index);
            blocks += w as u64 * h as u64;
        }
        Ok(blocks)
    }

    /// The coefficient storage the decoder retains for both baseline and
    /// progressive scans, refused above [`MAX_COEFFICIENT_BYTES`] before any
    /// allocation.
    fn coefficient_bytes(self) -> MediaResult<u64> {
        let bytes = self.blocks()? * 64 * size_of::<i16>() as u64;
        if bytes > MAX_COEFFICIENT_BYTES {
            return Err(MediaError::new(format!(
                "JPEG image {} x {} needs {bytes} bytes of coefficient storage, above the \
                 {MAX_COEFFICIENT_BYTES}-byte limit",
                self.width, self.height
            )));
        }
        Ok(bytes)
    }

    fn oriented_size(self) -> (u32, u32) {
        orientation::oriented_size(
            self.orientation,
            u32::from(self.width),
            u32::from(self.height),
        )
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
    let mut sampling = [(0, 0); 4];
    for (index, entry) in payload[6..].chunks_exact(3).enumerate() {
        let h = entry[1] >> 4;
        let v = entry[1] & 15;
        if seen[entry[0] as usize] || !(1..=4).contains(&h) || !(1..=4).contains(&v) || entry[2] > 3
        {
            return Err(MediaError::new("invalid JPEG component descriptor"));
        }
        seen[entry[0] as usize] = true;
        sampling[index] = (h, v);
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
        sampling,
    })
}

/// Exif APP1 segments kept before the first scan; later ones are dropped.
const EXIF_SEGMENTS: usize = 256;

/// The metadata a JPEG's APP1 segments carry before its first scan, kept
/// as Pillow's marker parser keeps it: the first `Exif` APP1 with every
/// later one's payload after its marker appended (up to
/// [`EXIF_SEGMENTS`]), and the last XMP APP1.
#[derive(Default)]
pub(crate) struct Metadata<'a> {
    exif: Vec<&'a [u8]>,
    xmp: Option<&'a [u8]>,
}

impl<'a> Metadata<'a> {
    pub(crate) fn app1(&mut self, payload: &'a [u8]) {
        if payload.starts_with(b"Exif\0\0") {
            if self.exif.len() < EXIF_SEGMENTS {
                self.exif.push(if self.exif.is_empty() {
                    payload
                } else {
                    &payload[6..]
                });
            }
        } else if let Some(xmp) = payload.strip_prefix(b"http://ns.adobe.com/xap/1.0/\0") {
            self.xmp = Some(xmp);
        }
    }

    /// The orientation the metadata gives, 1 when nothing transposes.
    pub(crate) fn orientation(&self) -> u8 {
        let exif =
            (!self.exif.is_empty()).then(|| orientation::exif(&orientation::Pieces(&self.exif)));
        orientation::resolve(exif, self.xmp.and_then(orientation::xmp))
    }
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
    let mut header = probe_header(bytes, limits)?;
    if policy == Orientation::Stored {
        header.orientation = 1;
    }
    Ok(header.oriented_size())
}

/// Decoder storage derived from the SOF, before any coefficient allocation.
/// Every component holds 64 i16 coefficients and 64 u8 samples per padded
/// block alongside the RGB output. Baseline and progressive scans both
/// retain full coefficient arrays. The fixed allowance covers component and
/// plane descriptors, scan state, Huffman tables, quantizers and IDCT scratch.
pub fn decode_storage_bytes(bytes: &[u8], limits: MediaLimits) -> MediaResult<u64> {
    let header = probe_header(bytes, limits)?;
    Ok(header.coefficient_bytes()?
        + header.blocks()? * 64
        + u64::from(header.width) * u64::from(header.height) * 3
        + 16 * 1024)
}

fn probe_header(bytes: &[u8], limits: MediaLimits) -> MediaResult<Header> {
    limits.check_bytes(bytes.len())?;
    let mut markers = Markers::new(bytes)?;
    let mut metadata = Metadata::default();
    let mut header: Option<Header> = None;
    loop {
        let (marker, payload) = markers.next()?;
        match marker {
            0xe1 => metadata.app1(payload),
            0xc0 | 0xc2 => {
                if header.is_some() {
                    return Err(MediaError::new("duplicate JPEG frame"));
                }
                let parsed = frame(marker, payload, 1)?;
                limits.check_named_image(
                    "JPEG",
                    u32::from(parsed.width),
                    u32::from(parsed.height),
                )?;
                parsed.coefficient_bytes()?;
                header = Some(parsed);
            }
            0xc1 | 0xc3..=0xcf if marker != 0xc4 && marker != 0xc8 && marker != 0xcc => {
                return Err(MediaError::new("unsupported JPEG frame coding"));
            }
            0xda => {
                let mut header = header.ok_or_else(|| MediaError::new("JPEG SOF is missing"))?;
                header.orientation = metadata.orientation();
                return Ok(header);
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
    fn storage_bound_counts_components_and_mcu_padding_in_both_scan_modes() {
        // A 17 x 9 source needs 2 x 1 MCUs at 4:2:0, and 3 x 2
        // blocks per full-resolution component at 4:4:4 or CMYK.
        for marker in [0xc0, 0xc2] {
            for (sampling, blocks) in [
                (vec![0x11], 6),
                (vec![0x11; 3], 18),
                (vec![0x22, 0x11, 0x11], 12),
                (vec![0x11; 4], 24),
            ] {
                let mut bytes = vec![0xff, 0xd8];
                let mut sof = vec![8, 0, 9, 0, 17, sampling.len() as u8];
                for (index, factors) in sampling.into_iter().enumerate() {
                    sof.extend_from_slice(&[index as u8, factors, 0]);
                }
                segment(&mut bytes, marker, &sof);
                segment(&mut bytes, 0xda, &[]);
                assert_eq!(
                    decode_storage_bytes(&bytes, limits()).unwrap(),
                    blocks * 64 * 3 + 17 * 9 * 3 + 16 * 1024,
                );
            }
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
    fn coefficient_limit_rejects_oversized_sof_before_allocation() {
        let mut bytes = simple_image(false);
        let sof = bytes
            .windows(2)
            .position(|pair| pair == [0xff, 0xc0])
            .unwrap();
        bytes[sof + 5..sof + 9].copy_from_slice(&[0xff, 0xff, 0xff, 0xff]);
        let mut permissive = limits();
        permissive.maximum_pixels = u64::MAX;
        // 65535 x 65535 gray needs 8 GiB of coefficients.
        let refusal = "JPEG image 65535 x 65535 needs 8589934592 bytes of coefficient storage, \
                       above the 805306368-byte limit";
        assert_eq!(probe(&bytes, permissive).unwrap_err().to_string(), refusal);
        assert_eq!(decode(&bytes, permissive).unwrap_err().to_string(), refusal);
        assert_eq!(
            decode_storage_bytes(&bytes, permissive)
                .unwrap_err()
                .to_string(),
            refusal
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
