//! Bounded PNG decoding. A chunk walker checks every chunk's CRC and place
//! and keeps the palette and the raw transparency chunk; the `png` crate
//! inflates and unfilters a stream of only the critical chunks, row by row,
//! and each row becomes RGB here. Under [`Orientation::Exif`] the walker
//! also reads orientation metadata as Pillow 12.3 does ([`Metadata`]), and
//! each pixel is written to its transposed position.

use std::io::Cursor;

use crate::core::{try_filled, try_with_capacity};
use crate::orientation::{self, Exif, Pieces, Placement};
use crate::{Alpha, MediaError, MediaLimits, MediaResult, Orientation, RgbImage};

const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
const IEND: &[u8; 12] = b"\0\0\0\0IEND\xae\x42\x60\x82";

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
        || !bytes.starts_with(SIGNATURE)
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
    if h.depth == 16 && matches!(h.color, 0 | 2 | 4 | 6) {
        return Err(MediaError::new("PNG with 16-bit samples is not supported"));
    }
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

/// The stored size from the IHDR, checked against `limits`.
pub fn probe(bytes: &[u8], limits: MediaLimits) -> MediaResult<(u32, u32)> {
    let h = header(bytes)?;
    limits.check_named_image("PNG", h.width, h.height)?;
    Ok((h.width, h.height))
}

/// The size under `orientation`: [`Orientation::Exif`] walks the chunks
/// (checking each CRC) for orientation metadata and transposes the size.
pub fn probe_oriented(
    bytes: &[u8],
    limits: MediaLimits,
    orientation: Orientation,
) -> MediaResult<(u32, u32)> {
    let (width, height) = probe(bytes, limits)?;
    if orientation == Orientation::Stored {
        return Ok((width, height));
    }
    limits.check_bytes(bytes.len())?;
    let chunks = chunks(bytes, header(bytes)?, orientation, false)?;
    Ok(orientation::oriented_size(
        chunks.orientation,
        width,
        height,
    ))
}

/// Decoder storage derived from the IHDR, before any allocation: the
/// critical-chunk stream (at most the source), the RGB image, and the
/// crate's working rows. Those are its unfiltering window of a few rows, at
/// least 128 KiB, the inflater's 32 KiB lookback and one output row, all at
/// the stored width; the fixed allowance also covers the inflater's tables
/// and the decoder state. Under [`Orientation::Exif`] it adds the largest
/// metadata value the walker holds at once ([`metadata_storage_bytes`]).
pub fn decode_storage_bytes(
    bytes: &[u8],
    limits: MediaLimits,
    orientation: Orientation,
) -> MediaResult<u64> {
    limits.check_bytes(bytes.len())?;
    let (width, height) = probe(bytes, limits)?;
    let h = header(bytes)?;
    let row = h.row_bytes(width as usize)? as u64 + 1;
    let metadata = if orientation == Orientation::Exif {
        metadata_storage_bytes(bytes)
    } else {
        0
    };
    (u64::from(width) * u64::from(height))
        .checked_mul(3)
        .and_then(|rgb| rgb.checked_add(row.checked_mul(16)?))
        .and_then(|sum| sum.checked_add(bytes.len() as u64 + (1 << 20) + metadata))
        .ok_or_else(|| MediaError::new("PNG decode size overflow"))
}

fn pass_size(size: u32, start: u32, step: u32) -> u32 {
    if size <= start {
        0
    } else {
        1 + (size - 1 - start) / step
    }
}

/// Adam7's passes as (x start, y start, x step, y step), the order the
/// crate returns their rows in.
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

pub(crate) fn composite(rgb: [u8; 3], alpha: u8, policy: Alpha) -> [u8; 3] {
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

/// What the chunk walker keeps: the palette, the raw transparency chunk
/// (the crate keeps only the low byte of a gray or RGB key), the stream the
/// crate decodes and the orientation the metadata gives.
struct Chunks {
    palette: Vec<[u8; 3]>,
    transparency: Vec<u8>,
    /// The signature, IHDR, an indexed image's PLTE, every IDAT and IEND,
    /// so no ancillary chunk can change what the crate decodes. Empty when
    /// the walk only probes.
    stream: Vec<u8>,
    orientation: u8,
}

/// Walks every chunk, refusing a CRC mismatch anywhere, a misplaced or
/// unknown critical chunk, data after IEND and an inconsistent chunk set.
/// `stream` keeps the critical chunks for the decoder; `orientation` reads
/// orientation metadata under [`Orientation::Exif`].
fn chunks(bytes: &[u8], h: Header, orientation: Orientation, stream: bool) -> MediaResult<Chunks> {
    let mut critical = Vec::new();
    if stream {
        critical = try_with_capacity(bytes.len())?;
        critical.extend_from_slice(&bytes[..33]);
    }
    let mut metadata = (orientation == Orientation::Exif).then(Metadata::default);
    let mut pos = 8_usize;
    let mut palette = Vec::new();
    let mut plte = 0..0;
    let mut transparency = Vec::new();
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
        if crc32fast::hash(&bytes[pos + 4..pos + 8 + len]) != be32(&bytes[pos + 8 + len..end]) {
            return Err(MediaError::new("PNG chunk CRC differs"));
        }
        if let Some(metadata) = &mut metadata {
            metadata.chunk(kind, data)?;
        }
        match kind {
            b"IHDR" if !seen_ihdr && pos == 8 && len == 13 => seen_ihdr = true,
            b"PLTE"
                if seen_ihdr && !seen_idat && len % 3 == 0 && len <= 768 && !data.is_empty() =>
            {
                palette = try_with_capacity(len / 3)?;
                palette.extend(data.chunks_exact(3).map(|rgb| [rgb[0], rgb[1], rgb[2]]));
                plte = pos..end;
            }
            b"tRNS" if seen_ihdr && !seen_idat => {
                transparency = try_with_capacity(len)?;
                transparency.extend_from_slice(data);
            }
            b"IDAT" if seen_ihdr => {
                if stream {
                    if !seen_idat && h.color == 3 {
                        critical.extend_from_slice(&bytes[plte.clone()]);
                    }
                    critical.extend_from_slice(&bytes[pos..end]);
                }
                seen_idat = true;
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
    if stream {
        critical.extend_from_slice(IEND);
    }
    Ok(Chunks {
        palette,
        transparency,
        stream: critical,
        orientation: metadata.map_or(1, |metadata| metadata.orientation()),
    })
}

/// Pillow's limit on one decompressed text chunk (`MAX_TEXT_CHUNK`): past
/// it Pillow refuses the image, and so does the walker for the chunks it
/// reads.
const TEXT_LIMIT: usize = 1 << 20;

/// The bytes the walker may inflate and hex-decode for orientation in one
/// image, repeated keys included, before it refuses the image. Pillow's
/// own limit on all text is 64 MiB; this bounds the work a probe does
/// before the host reserves anything.
const METADATA_BUDGET: u64 = 4 << 20;

/// The text keys Pillow reads orientation from.
const EXIF_KEY: &[u8] = b"exif";
const RAW_PROFILE_KEY: &[u8] = b"Raw profile type exif";
const XMP_KEY: &[u8] = b"XML:com.adobe.xmp";

/// The orientation metadata of a PNG as Pillow 12.3 collects it: chunks
/// before the first IDAT, then after it up to IEND, or for an animated PNG
/// up to the next frame's fcTL (`PngImageFile.load_end`). Each Pillow info
/// key keeps its last value, and each value is read when its chunk is, so
/// one inflated chunk is held at a time:
///
/// - `exif`: an eXIf chunk or a tEXt keyed `exif` (bytes). A zTXt or iTXt
///   keyed `exif` stores text, which Pillow cannot read (it raises) unless
///   it is empty.
/// - `Raw profile type exif` (tEXt, zTXt or iTXt): hex after three lines,
///   read only without an `exif` value; invalid hex makes Pillow raise.
/// - XMP: `XML:com.adobe.xmp` text (tEXt, zTXt, or an iTXt that decodes as
///   UTF-8), else the bytes of the last iTXt with that key.
///
/// A zTXt with corrupt zlib data empties its key; an iTXt with corrupt or
/// unknown compression is ignored, and a zTXt with an unknown compression
/// method (which Pillow refuses) too. Truncated zlib data reads as far as
/// it goes, as zlib's streaming decompressor does.
#[derive(Default)]
struct Metadata {
    exif: Option<Exif>,
    raw_profile: Option<Exif>,
    /// The XMP digit of a non-empty `XML:com.adobe.xmp` text, or None
    /// when that is unset or empty.
    xmp_text: Option<Option<u8>>,
    /// The same for the last iTXt's XMP bytes.
    xmp_bytes: Option<Option<u8>>,
    frames: Option<u32>,
    fctl_before_idat: bool,
    seen_idat: bool,
    animated: bool,
    done: bool,
    spent: u64,
}

/// A text chunk's value as the walker reads it.
enum Value<'a> {
    Bytes(&'a [u8]),
    Inflated(Vec<u8>),
}

impl Value<'_> {
    fn bytes(&self) -> &[u8] {
        match self {
            Value::Bytes(bytes) => bytes,
            Value::Inflated(bytes) => bytes,
        }
    }
}

impl Metadata {
    fn chunk(&mut self, kind: &[u8], data: &[u8]) -> MediaResult<()> {
        if self.done {
            return Ok(());
        }
        match kind {
            b"acTL" if !self.seen_idat && data.len() >= 8 => {
                // A second acTL invalidates the animation, as Pillow's does.
                self.frames = match self.frames {
                    Some(_) => None,
                    None => Some(be32(&data[..4])).filter(|&n| n != 0 && n <= 0x8000_0000),
                };
            }
            b"fcTL" if !self.seen_idat => self.fctl_before_idat = true,
            b"fcTL" if self.animated => self.done = true,
            b"IDAT" if !self.seen_idat => {
                self.seen_idat = true;
                let default_image = self.frames.is_some() && !self.fctl_before_idat;
                self.animated = u64::from(self.frames.unwrap_or(1)) + u64::from(default_image) > 1;
            }
            b"IEND" => self.done = true,
            b"eXIf" => self.exif = Some(orientation::exif(&Pieces(&[data]))),
            b"tEXt" => {
                let (key, value) = split_nul(data);
                self.text(key, Value::Bytes(value), true)?;
            }
            b"zTXt" => {
                let (key, value) = split_nul(data);
                if !is_orientation_key(key) || value.first().is_some_and(|&method| method != 0) {
                    return Ok(());
                }
                let value = match self.inflate(value.get(1..).unwrap_or_default())? {
                    Some(inflated) => Value::Inflated(inflated),
                    None => Value::Bytes(&[]),
                };
                self.text(key, value, false)?;
            }
            b"iTXt" => {
                let Some(nul) = data.iter().position(|&byte| byte == 0) else {
                    return Ok(());
                };
                let (key, rest) = (&data[..nul], &data[nul + 1..]);
                if !is_orientation_key(key) || rest.len() < 2 {
                    return Ok(());
                }
                let (compressed, method, rest) = (rest[0], rest[1], &rest[2..]);
                let mut fields = rest.splitn(3, |&byte| byte == 0);
                let (Some(language), Some(translated), Some(value)) =
                    (fields.next(), fields.next(), fields.next())
                else {
                    return Ok(());
                };
                let value = match (compressed, method) {
                    (0, _) => Value::Bytes(value),
                    (_, 0) => match self.inflate(value)? {
                        Some(inflated) => Value::Inflated(inflated),
                        None => return Ok(()),
                    },
                    _ => return Ok(()),
                };
                if key == XMP_KEY {
                    self.xmp_bytes = nonempty_xmp(value.bytes());
                }
                let utf8 = |bytes: &[u8]| std::str::from_utf8(bytes).is_ok();
                if utf8(language) && utf8(translated) && utf8(value.bytes()) {
                    self.text(key, value, false)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Records a text value under `key`; `raw` is a tEXt's, whose `exif`
    /// value stays bytes.
    fn text(&mut self, key: &[u8], value: Value<'_>, raw: bool) -> MediaResult<()> {
        let bytes = value.bytes();
        match key {
            EXIF_KEY if raw => self.exif = Some(orientation::exif(&Pieces(&[bytes]))),
            EXIF_KEY if bytes.is_empty() => self.exif = Some(Exif::Absent),
            EXIF_KEY => self.exif = Some(Exif::Unreadable),
            RAW_PROFILE_KEY => self.raw_profile = Some(self.read_raw_profile(bytes)?),
            XMP_KEY => self.xmp_text = nonempty_xmp(bytes),
            _ => {}
        }
        Ok(())
    }

    /// Reads a raw profile: hex after its first three lines, byte pairs
    /// separated by ASCII whitespace at most, as `bytes.fromhex` takes them.
    fn read_raw_profile(&mut self, value: &[u8]) -> MediaResult<Exif> {
        let mut lines = 0;
        let start = value
            .iter()
            .position(|&byte| {
                lines += usize::from(byte == b'\n');
                lines == 3
            })
            .map_or(value.len(), |end| end + 1);
        let digits = || value[start..].iter().copied().filter(|&byte| byte != b'\n');
        let mut count = 0_u64;
        let mut pending = false;
        for byte in digits() {
            if byte.is_ascii_hexdigit() {
                count += u64::from(pending);
                pending = !pending;
            } else if pending || !matches!(byte, b' ' | b'\t' | b'\r' | 0x0b | 0x0c) {
                return Ok(Exif::Unreadable);
            }
        }
        if pending {
            return Ok(Exif::Unreadable);
        }
        self.spend(count)?;
        let mut decoded = try_with_capacity(count as usize)?;
        let mut high = None;
        for byte in digits().filter(u8::is_ascii_hexdigit) {
            let nibble = (byte as char).to_digit(16).expect("hex digit") as u8;
            match high.take() {
                Some(high) => decoded.push((high << 4) | nibble),
                None => high = Some(nibble),
            }
        }
        Ok(orientation::exif(&Pieces(&[&decoded])))
    }

    /// Inflates zlib data as zlib's streaming decompressor does, up to
    /// [`TEXT_LIMIT`]: truncated data gives what it holds, corrupt data
    /// None, and more than the limit refuses the image.
    fn inflate(&mut self, data: &[u8]) -> MediaResult<Option<Vec<u8>>> {
        use miniz_oxide::inflate::TINFLStatus;
        use miniz_oxide::inflate::core::{DecompressorOxide, decompress, inflate_flags};
        let flags = inflate_flags::TINFL_FLAG_PARSE_ZLIB_HEADER
            | inflate_flags::TINFL_FLAG_COMPUTE_ADLER32
            | inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
        let mut state = DecompressorOxide::new();
        let mut out = try_filled(data.len().saturating_mul(2).clamp(64, TEXT_LIMIT + 1), 0_u8)?;
        let (mut input, mut written) = (data, 0);
        let result = loop {
            let (status, read, wrote) = decompress(&mut state, input, &mut out, written, flags);
            input = &input[read..];
            written += wrote;
            match status {
                TINFLStatus::Done | TINFLStatus::FailedCannotMakeProgress => break Some(()),
                TINFLStatus::HasMoreOutput if out.len() <= TEXT_LIMIT => {
                    let (len, grown) = (out.len(), (out.len() * 2).min(TEXT_LIMIT + 1));
                    crate::core::try_reserve(&mut out, grown - len)?;
                    out.resize(grown, 0);
                }
                TINFLStatus::HasMoreOutput => {
                    return Err(MediaError::new(format!(
                        "PNG compressed text inflates past {TEXT_LIMIT} bytes"
                    )));
                }
                _ => break None,
            }
        };
        if written > TEXT_LIMIT {
            return Err(MediaError::new(format!(
                "PNG compressed text inflates past {TEXT_LIMIT} bytes"
            )));
        }
        self.spend(written as u64)?;
        Ok(result.map(|()| {
            out.truncate(written);
            out
        }))
    }

    fn spend(&mut self, bytes: u64) -> MediaResult<()> {
        self.spent += bytes;
        if self.spent > METADATA_BUDGET {
            return Err(MediaError::new(format!(
                "PNG orientation metadata inflates past {METADATA_BUDGET} bytes"
            )));
        }
        Ok(())
    }

    fn orientation(&self) -> u8 {
        // A non-empty text wins even without a digit; an empty one falls
        // back to the iTXt bytes.
        let xmp = self.xmp_text.or(self.xmp_bytes).flatten();
        orientation::resolve(self.exif.or(self.raw_profile), xmp)
    }
}

fn split_nul(data: &[u8]) -> (&[u8], &[u8]) {
    match data.iter().position(|&byte| byte == 0) {
        Some(nul) => (&data[..nul], &data[nul + 1..]),
        None => (data, &[]),
    }
}

fn is_orientation_key(key: &[u8]) -> bool {
    matches!(key, EXIF_KEY | RAW_PROFILE_KEY | XMP_KEY)
}

/// The XMP digit of a value, as Some(None) for a non-empty value without
/// one, or None for an empty value (which Pillow treats as unset).
fn nonempty_xmp(value: &[u8]) -> Option<Option<u8>> {
    (!value.is_empty()).then(|| orientation::xmp(value))
}

/// The most metadata storage the walker holds at once, from chunk headers
/// alone: one inflated chunk of at most [`TEXT_LIMIT`] and one decoded raw
/// profile of at most half its text, within [`METADATA_BUDGET`].
fn metadata_storage_bytes(bytes: &[u8]) -> u64 {
    let mut pos = 8_usize;
    let mut largest = 0_u64;
    while pos + 12 <= bytes.len() {
        let len = be32(&bytes[pos..pos + 4]) as usize;
        let Some(end) = pos.checked_add(12).and_then(|n| n.checked_add(len)) else {
            break;
        };
        if end > bytes.len() {
            break;
        }
        let kind = &bytes[pos + 4..pos + 8];
        let data = &bytes[pos + 8..pos + 8 + len];
        let key = split_nul(data).0;
        if matches!(kind, b"tEXt" | b"zTXt" | b"iTXt") && is_orientation_key(key) {
            let compressed = kind != b"tEXt";
            let text = if compressed {
                TEXT_LIMIT as u64
            } else {
                len as u64
            };
            let profile = if key == RAW_PROFILE_KEY {
                (text / 2).min(METADATA_BUDGET)
            } else {
                0
            };
            let inflated = if compressed { TEXT_LIMIT as u64 + 1 } else { 0 };
            largest = largest.max(inflated + profile);
        }
        pos = end;
    }
    largest
}

fn invalid(error: ::png::DecodingError) -> MediaError {
    MediaError::new(format!("PNG image data is invalid: {error}"))
}

/// Decodes with transparency composited onto white.
pub fn decode(bytes: &[u8], limits: MediaLimits) -> MediaResult<RgbImage> {
    decode_with_alpha(bytes, limits, Alpha::CompositeWhite)
}

/// Decodes with transparency (an alpha channel or a tRNS chunk) handled
/// as `alpha` says, in the stored orientation.
pub fn decode_with_alpha(bytes: &[u8], limits: MediaLimits, alpha: Alpha) -> MediaResult<RgbImage> {
    decode_oriented(bytes, limits, alpha, Orientation::Stored)
}

/// [`decode_with_alpha`] under `orientation`: [`Orientation::Exif`]
/// transposes the image as its orientation metadata says ([`Metadata`]).
pub fn decode_oriented(
    bytes: &[u8],
    limits: MediaLimits,
    alpha: Alpha,
    orientation: Orientation,
) -> MediaResult<RgbImage> {
    limits.check_bytes(bytes.len())?;
    let h = header(bytes)?;
    limits.check_named_image("PNG", h.width, h.height)?;
    let chunks = chunks(bytes, h, orientation, true)?;
    let placement = Placement::new(chunks.orientation, h.width, h.height);
    let mut options = ::png::DecodeOptions::default();
    options.set_ignore_adler32(false);
    let mut decoder =
        ::png::Decoder::new_with_options(Cursor::new(chunks.stream.as_slice()), options);
    // The walker has bounded the stream and the size; the crate's own limit
    // would refuse wide rows the bound already counts.
    decoder.set_limits(::png::Limits { bytes: usize::MAX });
    decoder.set_transformations(::png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().map_err(invalid)?;
    let output_len = usize::try_from(
        (u64::from(h.width) * u64::from(h.height))
            .checked_mul(3)
            .ok_or_else(|| MediaError::new("PNG pixel size overflow"))?,
    )
    .map_err(|_| MediaError::new("PNG pixel size overflow"))?;
    let mut rgb = try_filled(output_len, 0_u8)?;
    for &(sx, sy, dx, dy) in passes(h.interlace) {
        let pass_width = pass_size(h.width, sx, dx) as usize;
        let pass_height = pass_size(h.height, sy, dy) as usize;
        if pass_width == 0 || pass_height == 0 {
            continue;
        }
        let stride = h.row_bytes(pass_width)?;
        for y in 0..pass_height {
            let row = reader
                .next_row()
                .map_err(invalid)?
                .ok_or_else(|| MediaError::new("PNG image data ends early"))?;
            let line = row.data();
            if line.len() != stride {
                return Err(MediaError::new("PNG row size differs from its IHDR"));
            }
            let row = sy as usize + y * dy as usize;
            for x in 0..pass_width {
                let target = placement.index(sx as usize + x * dx as usize, row) * 3;
                rgb[target..target + 3].copy_from_slice(&pixel(
                    line,
                    x,
                    h,
                    &chunks.palette,
                    &chunks.transparency,
                    alpha,
                )?);
            }
        }
    }
    if reader.next_row().map_err(invalid)?.is_some() {
        return Err(MediaError::new("PNG has more rows than its IHDR"));
    }
    reader.finish().map_err(invalid)?;
    let (width, height) = placement.size();
    RgbImage::new(width, height, rgb, limits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(kind: &[u8; 4], payload: &[u8], output: &mut Vec<u8>) {
        output.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        output.extend_from_slice(kind);
        output.extend_from_slice(payload);
        let start = output.len() - payload.len() - 4;
        output.extend_from_slice(&crc32fast::hash(&output[start..]).to_be_bytes());
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

    fn png(ihdr: [u8; 13], chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let mut png = SIGNATURE.to_vec();
        chunk(b"IHDR", &ihdr, &mut png);
        for (kind, payload) in chunks {
            chunk(kind, payload, &mut png);
        }
        png
    }

    fn ihdr(width: u32, height: u32, depth: u8, color: u8, interlace: u8) -> [u8; 13] {
        let mut ihdr = [0; 13];
        ihdr[..4].copy_from_slice(&width.to_be_bytes());
        ihdr[4..8].copy_from_slice(&height.to_be_bytes());
        ihdr[8..].copy_from_slice(&[depth, color, 0, 0, interlace]);
        ihdr
    }

    #[test]
    fn large_rgba_allocation_failures_return_media_errors() {
        let (width, height) = (8192_u32, 512_u32);
        let stride = width as usize * 4;
        let raw = vec![0_u8; (stride + 1) * height as usize];
        let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&raw, 6);
        let png = png(
            ihdr(width, height, 8, 6, 0),
            &[(b"IDAT", &compressed), (b"IEND", &[])],
        );
        let limits = MediaLimits {
            maximum_pixels: 1 << 24,
            ..limits()
        };
        let decode = || decode_with_alpha(&png, limits, Alpha::Discard);
        assert_eq!(
            decode().unwrap().data.len(),
            width as usize * height as usize * 3
        );
        // The critical-chunk stream and the RGB image must report a fallible
        // reservation failure instead of aborting.
        for size in [png.len(), width as usize * height as usize * 3] {
            let result = crate::core::with_allocation_failure(size, 0, decode);
            assert_eq!(
                result.unwrap_err().to_string(),
                "media buffer allocation failed",
                "{size}"
            );
        }
    }

    #[test]
    fn adam7_palette_alpha_composites_on_white() {
        let png = png(
            ihdr(1, 1, 1, 3, 1),
            &[
                (b"PLTE", &[0, 0, 255, 255, 0, 0]),
                (b"tRNS", &[128, 255]),
                (
                    b"IDAT",
                    &miniz_oxide::deflate::compress_to_vec_zlib(&[0, 0], 6),
                ),
                (b"IEND", &[]),
            ],
        );
        assert_eq!(decode(&png, limits()).unwrap().data, [127, 127, 255]);
        let discarded = decode_with_alpha(&png, limits(), Alpha::Discard).unwrap();
        assert_eq!(discarded.data, [0, 0, 255]);
        let mut bad = png;
        bad[29] ^= 1;
        assert!(decode(&bad, limits()).is_err());
    }

    /// A 16-bit tRNS key below 16-bit samples matches only when its high
    /// byte is zero; the crate would keep its low byte alone.
    #[test]
    fn gray_key_uses_the_whole_trns_value() {
        // Two 4-bit samples, 4 and 0.
        let idat = miniz_oxide::deflate::compress_to_vec_zlib(&[0, 0x40], 6);
        for (key, expected) in [([0, 4], [255; 3]), ([1, 4], [0x44; 3])] {
            let png = png(
                ihdr(2, 1, 4, 0, 0),
                &[(b"tRNS", &key), (b"IDAT", &idat), (b"IEND", &[])],
            );
            let image = decode(&png, limits()).unwrap();
            assert_eq!(image.data[..3], expected, "{key:?}");
            assert_eq!(image.data[3..], [0; 3], "{key:?}");
        }
    }

    /// Ancillary chunks never reach the crate: an APNG frame control that
    /// names a smaller frame leaves the IHDR image as decoded.
    #[test]
    fn ancillary_chunks_do_not_change_the_image() {
        let idat = miniz_oxide::deflate::compress_to_vec_zlib(&[0, 1, 2, 3, 0, 4, 5, 6], 6);
        let plain = png(ihdr(1, 2, 8, 2, 0), &[(b"IDAT", &idat), (b"IEND", &[])]);
        let mut fctl = [0_u8; 26];
        fctl[4..8].copy_from_slice(&1_u32.to_be_bytes());
        fctl[8..12].copy_from_slice(&1_u32.to_be_bytes());
        let animated = png(
            ihdr(1, 2, 8, 2, 0),
            &[
                (b"acTL", &[0, 0, 0, 1, 0, 0, 0, 0]),
                (b"fcTL", &fctl),
                (b"eXIf", b"not TIFF"),
                (b"IDAT", &idat),
                (b"IEND", &[]),
            ],
        );
        let expected = decode(&plain, limits()).unwrap();
        assert_eq!(expected.data, [1, 2, 3, 4, 5, 6]);
        assert_eq!(decode(&animated, limits()).unwrap(), expected);
    }

    /// Image data whose zlib checksum is wrong, that inflates short, or
    /// that follows IEND is refused, as the walker and inflater did before.
    #[test]
    fn checksums_truncation_and_trailing_data_are_refused() {
        let mut idat = miniz_oxide::deflate::compress_to_vec_zlib(&[0, 1, 2, 3], 6);
        let good = png(ihdr(1, 1, 8, 2, 0), &[(b"IDAT", &idat), (b"IEND", &[])]);
        assert_eq!(decode(&good, limits()).unwrap().data, [1, 2, 3]);
        let mut trailing = good.clone();
        trailing.push(0);
        assert!(decode(&trailing, limits()).is_err());
        let short = miniz_oxide::deflate::compress_to_vec_zlib(&[0, 1, 2], 6);
        let short = png(ihdr(1, 1, 8, 2, 0), &[(b"IDAT", &short), (b"IEND", &[])]);
        assert!(decode(&short, limits()).is_err());
        let last = idat.len() - 1;
        idat[last] ^= 1;
        let adler = png(ihdr(1, 1, 8, 2, 0), &[(b"IDAT", &idat), (b"IEND", &[])]);
        assert!(decode(&adler, limits()).is_err());
        for cut in 0..good.len() {
            assert!(decode(&good[..cut], limits()).is_err(), "{cut}");
        }
    }

    #[test]
    fn storage_bound_refuses_header_overflow() {
        // A readable header is enough to reserve before decoding. Even
        // permissive callers must not wrap its storage calculation.
        let png = png(ihdr(u32::MAX, u32::MAX, 8, 6, 0), &[]);
        let limits = MediaLimits {
            maximum_pixels: u64::MAX,
            ..limits()
        };
        assert_eq!(
            decode_storage_bytes(&png, limits, Orientation::Stored)
                .unwrap_err()
                .to_string(),
            "PNG decode size overflow"
        );
    }

    #[test]
    fn sixteen_bit_and_oversized_images_name_the_reason() {
        let wide = png(ihdr(2, 1, 16, 2, 0), &[(b"IEND", &[])]);
        assert_eq!(
            probe(&wide, limits()).unwrap_err().to_string(),
            "PNG with 16-bit samples is not supported"
        );
        let large = png(ihdr(2048, 1024, 8, 2, 0), &[(b"IEND", &[])]);
        assert_eq!(
            probe(&large, limits()).unwrap_err().to_string(),
            "PNG image 2048 x 1024 has 2097152 pixels, above the 1048576-pixel limit"
        );
    }
}
