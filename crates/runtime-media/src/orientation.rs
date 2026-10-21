//! Image orientation as Pillow 12.3's `ImageOps.exif_transpose` reads it:
//! the EXIF orientation tag (0x0112) of IFD0, else an XMP
//! `tiff:Orientation`, wherever a format carries them. The decoders gather
//! a format's EXIF and XMP blocks; this module reads them and transposes.
//!
//! EXIF is read as Pillow's TIFF reader does. Leading `Exif\0\0` markers
//! are skipped; the stream must start with one of the four classic TIFF
//! byte-order headers. IFD0's entries are read until the data runs out, and
//! the last orientation entry wins. Its first value counts when its type is
//! a number Pillow registers (SHORT, LONG, SBYTE, SSHORT, SLONG, RATIONAL,
//! SRATIONAL, FLOAT, DOUBLE, IFD or LONG8); BYTE, ASCII and UNDEFINED
//! entries are present but never a number, and other types are skipped.
//! Only an exact integer from 2 to 8 transposes. XMP is consulted only when
//! IFD0 has no orientation entry at all: its first `tiff:Orientation="d` or
//! `tiff:Orientation>d` digit.

/// What an EXIF block says about orientation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Exif {
    /// IFD0 holds no orientation entry, so XMP may supply one.
    Absent,
    /// An orientation entry, with its value when that is an exact integer.
    Present(Option<i64>),
    /// Pillow cannot read the block: it raises, or for a JPEG keeps an
    /// empty EXIF. Either way nothing transposes, XMP included.
    Unreadable,
}

/// EXIF data as one byte stream over one or more pieces (a JPEG's APP1
/// payloads are read as one), without copying them.
pub(crate) struct Pieces<'a>(pub &'a [&'a [u8]]);

impl Pieces<'_> {
    fn len(&self) -> usize {
        self.0.iter().map(|piece| piece.len()).sum()
    }

    /// Copies `out.len()` bytes from `offset`, or false past the end.
    fn copy(&self, mut offset: usize, out: &mut [u8]) -> bool {
        let mut filled = 0;
        for piece in self.0 {
            if offset >= piece.len() {
                offset -= piece.len();
                continue;
            }
            let take = (piece.len() - offset).min(out.len() - filled);
            out[filled..filled + take].copy_from_slice(&piece[offset..offset + take]);
            filled += take;
            offset = 0;
            if filled == out.len() {
                return true;
            }
        }
        filled == out.len()
    }
}

const ORIENTATION: u16 = 0x0112;

/// Reads the orientation entry of an EXIF block.
pub(crate) fn exif(data: &Pieces<'_>) -> Exif {
    let total = data.len();
    let mut start = 0;
    let mut marker = [0_u8; 6];
    while start + 6 <= total && data.copy(start, &mut marker) && &marker == b"Exif\0\0" {
        start += 6;
    }
    if start == total {
        return Exif::Absent;
    }
    let mut header = [0_u8; 8];
    if !data.copy(start, &mut header) {
        return Exif::Unreadable;
    }
    let little = match &header[..4] {
        b"II*\0" | b"II\0*" => true,
        b"MM\0*" | b"MM*\0" => false,
        _ => return Exif::Unreadable,
    };
    let at = |offset: usize, out: &mut [u8]| {
        offset
            .checked_add(start)
            .is_some_and(|offset| data.copy(offset, out))
    };
    let u16_at = |bytes: [u8; 2]| {
        if little {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        }
    };
    let u32_at = |bytes: [u8; 4]| {
        if little {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        }
    };
    let ifd = u32_at(header[4..8].try_into().expect("four bytes")) as usize;
    let mut count = [0_u8; 2];
    if !at(ifd, &mut count) {
        return Exif::Absent;
    }
    let mut found = Exif::Absent;
    for index in 0..usize::from(u16_at(count)) {
        let mut entry = [0_u8; 12];
        if !at(ifd + 2 + index * 12, &mut entry) {
            break;
        }
        let kind = u16_at([entry[2], entry[3]]);
        let Some(unit) = unit_size(kind) else {
            continue;
        };
        let size = u64::from(u32_at(entry[4..8].try_into().expect("four bytes"))) * unit;
        if size == 0 {
            continue;
        }
        // The first value is all that counts, at most eight bytes.
        let mut first = [0_u8; 8];
        let width = unit as usize;
        if size <= 4 {
            first[..width].copy_from_slice(&entry[8..8 + width]);
        } else {
            // A value past the end stops the IFD there, whatever its tag.
            let offset = u64::from(u32_at(entry[8..12].try_into().expect("four bytes")));
            if offset + size > (total - start) as u64 || !at(offset as usize, &mut first[..width]) {
                break;
            }
        }
        if u16_at([entry[0], entry[1]]) == ORIENTATION {
            found = Exif::Present(number(kind, &first, little));
        }
    }
    found
}

/// The byte size of one value of a TIFF type Pillow reads, or None for a
/// type it skips.
fn unit_size(kind: u16) -> Option<u64> {
    Some(match kind {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 | 16 => 8,
        _ => return None,
    })
}

/// The first value of an entry as an exact integer, or None when its type
/// is not a number or the value is not an integer.
fn number(kind: u16, bytes: &[u8; 8], little: bool) -> Option<i64> {
    let get = |width: usize| -> u64 {
        let mut value = 0_u64;
        for index in 0..width {
            let byte = bytes[if little { width - 1 - index } else { index }];
            value = (value << 8) | u64::from(byte);
        }
        value
    };
    let ratio = |numerator: i64, denominator: i64| {
        (denominator != 0 && numerator % denominator == 0).then(|| numerator / denominator)
    };
    let float = |value: f64| (value.fract() == 0.0 && value.abs() < 1e18).then_some(value as i64);
    match kind {
        3 => Some(get(2) as i64),
        4 | 13 => Some(get(4) as i64),
        16 => i64::try_from(get(8)).ok(),
        6 => Some(i64::from(bytes[0] as i8)),
        8 => Some(i64::from(get(2) as u16 as i16)),
        9 => Some(i64::from(get(4) as u32 as i32)),
        5 => {
            let (numerator, denominator) = split(bytes, little);
            ratio(i64::from(numerator), i64::from(denominator))
        }
        10 => {
            let (numerator, denominator) = split(bytes, little);
            ratio(i64::from(numerator as i32), i64::from(denominator as i32))
        }
        11 => float(f64::from(f32::from_bits(get(4) as u32))),
        12 => float(f64::from_bits(get(8))),
        _ => None,
    }
}

/// A RATIONAL's numerator and denominator.
fn split(bytes: &[u8; 8], little: bool) -> (u32, u32) {
    let word = |part: &[u8]| {
        let part: [u8; 4] = part.try_into().expect("four bytes");
        if little {
            u32::from_le_bytes(part)
        } else {
            u32::from_be_bytes(part)
        }
    };
    (word(&bytes[..4]), word(&bytes[4..]))
}

/// The first `tiff:Orientation` digit of an XMP packet.
pub(crate) fn xmp(data: &[u8]) -> Option<u8> {
    const NAME: &[u8] = b"tiff:Orientation";
    let mut position = 0;
    while let Some(found) = find(&data[position..], NAME) {
        let after = position + found + NAME.len();
        let digit = match data.get(after..) {
            Some([b'=', b'"', digit, ..]) | Some([b'>', digit, ..]) => *digit,
            _ => 0,
        };
        if digit.is_ascii_digit() {
            return Some(digit - b'0');
        }
        position = position + found + 1;
    }
    None
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// The orientation an image takes from its EXIF block (`exif`, read with
/// [`exif`]) and its XMP digit, 1 when nothing transposes.
pub(crate) fn resolve(exif: Option<Exif>, xmp: Option<u8>) -> u8 {
    let value = match exif.unwrap_or(Exif::Absent) {
        Exif::Unreadable => return 1,
        Exif::Present(value) => value,
        Exif::Absent => xmp.map(i64::from),
    };
    match value {
        Some(value @ 2..=8) => value as u8,
        _ => 1,
    }
}

/// The size of a `width` x `height` image once transposed by `orientation`.
pub(crate) fn oriented_size(orientation: u8, width: u32, height: u32) -> (u32, u32) {
    if matches!(orientation, 5..=8) {
        (height, width)
    } else {
        (width, height)
    }
}

/// The stored position of output pixel (`x`, `y`) of a `width` x `height`
/// stored image transposed by `orientation`.
pub(crate) fn source_position(
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

/// Where stored pixels land once transposed by an orientation: the inverse
/// of [`source_position`], for decoders that write as they decode.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Placement {
    orientation: u8,
    width: usize,
    height: usize,
}

impl Placement {
    /// For a stored `width` x `height` image.
    pub(crate) fn new(orientation: u8, width: u32, height: u32) -> Self {
        Self {
            orientation,
            width: width as usize,
            height: height as usize,
        }
    }

    /// The output size.
    pub(crate) fn size(self) -> (u32, u32) {
        oriented_size(self.orientation, self.width as u32, self.height as u32)
    }

    /// The output pixel index of stored pixel (`x`, `y`).
    pub(crate) fn index(self, x: usize, y: usize) -> usize {
        let (w, h) = (self.width, self.height);
        let (out_x, out_y, out_width) = match self.orientation {
            2 => (w - 1 - x, y, w),
            3 => (w - 1 - x, h - 1 - y, w),
            4 => (x, h - 1 - y, w),
            5 => (y, x, h),
            6 => (h - 1 - y, x, h),
            7 => (h - 1 - y, w - 1 - x, h),
            8 => (y, w - 1 - x, h),
            _ => (x, y, w),
        };
        out_y * out_width + out_x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(bytes: &[u8]) -> Exif {
        exif(&Pieces(&[bytes]))
    }

    fn ifd(entries: &[(u16, u16, u32, [u8; 4])]) -> Vec<u8> {
        let mut out = b"II*\0\x08\0\0\0".to_vec();
        out.extend((entries.len() as u16).to_le_bytes());
        for (tag, kind, count, value) in entries {
            out.extend(tag.to_le_bytes());
            out.extend(kind.to_le_bytes());
            out.extend(count.to_le_bytes());
            out.extend(value);
        }
        out.extend([0; 4]);
        out
    }

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
    fn placement_inverts_source_position() {
        let (width, height) = (3, 2);
        for orientation in 1..=8 {
            let placement = Placement::new(orientation, width, height);
            let (out_width, out_height) = placement.size();
            for y in 0..out_height as usize {
                for x in 0..out_width as usize {
                    let (sx, sy) =
                        source_position(orientation, x, y, width as usize, height as usize);
                    assert_eq!(
                        placement.index(sx, sy),
                        y * out_width as usize + x,
                        "{orientation}"
                    );
                }
            }
        }
    }

    #[test]
    fn exif_values_follow_the_tiff_reader() {
        let short = |value: u16| [value.to_le_bytes()[0], value.to_le_bytes()[1], 0, 0];
        assert_eq!(
            read(&ifd(&[(0x112, 3, 1, short(6))])),
            Exif::Present(Some(6))
        );
        // The last entry wins, BYTE is never a number and an unknown type
        // is skipped.
        let entries = [(0x112, 3, 1, short(6)), (0x112, 3, 1, short(3))];
        assert_eq!(read(&ifd(&entries)), Exif::Present(Some(3)));
        assert_eq!(
            read(&ifd(&[(0x112, 1, 1, [6, 0, 0, 0])])),
            Exif::Present(None)
        );
        assert_eq!(read(&ifd(&[(0x112, 17, 1, [6, 0, 0, 0])])), Exif::Absent);
        assert_eq!(read(&ifd(&[(0x112, 3, 0, short(6))])), Exif::Absent);
        // Leading markers are skipped; a bad header cannot be read.
        let marked = [
            b"Exif\0\0Exif\0\0".as_slice(),
            &ifd(&[(0x112, 3, 1, short(8))]),
        ]
        .concat();
        assert_eq!(read(&marked), Exif::Present(Some(8)));
        assert_eq!(read(b"Exif\0\0"), Exif::Absent);
        assert_eq!(read(b"XX*\0\x08\0\0\0"), Exif::Unreadable);
        assert_eq!(read(b"II*\0"), Exif::Unreadable);
        // A value split across pieces reads as one stream.
        let whole = ifd(&[(0x112, 3, 1, short(5))]);
        let (head, tail) = whole.split_at(13);
        assert_eq!(exif(&Pieces(&[head, tail])), Exif::Present(Some(5)));
    }

    #[test]
    fn xmp_takes_the_first_digit_and_only_without_an_exif_entry() {
        assert_eq!(xmp(br#"<a tiff:Orientation="6"/>"#), Some(6));
        assert_eq!(xmp(b"<tiff:Orientation>7</tiff:Orientation>"), Some(7));
        assert_eq!(
            xmp(br#"tiff:Orientation='6' tiff:Orientation="3""#),
            Some(3)
        );
        assert_eq!(xmp(b"tiff:Orientation"), None);
        assert_eq!(resolve(Some(Exif::Present(Some(1))), Some(6)), 1);
        assert_eq!(resolve(Some(Exif::Present(None)), Some(6)), 1);
        assert_eq!(resolve(Some(Exif::Unreadable), Some(6)), 1);
        assert_eq!(resolve(Some(Exif::Absent), Some(6)), 6);
        assert_eq!(resolve(None, Some(9)), 1);
    }
}
