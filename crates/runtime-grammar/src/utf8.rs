//! Unicode scalar ranges as UTF-8 byte-range sequences.
//!
//! A range of scalar values is split until every piece encodes to one
//! sequence of byte ranges, so a byte automaton built from the pieces accepts
//! exactly the UTF-8 encodings of the range and nothing else.

/// One run of byte ranges; each position accepts one byte in its range.
pub(crate) type ByteRanges = Vec<(u8, u8)>;

const SURROGATES: (u32, u32) = (0xD800, 0xDFFF);
const MAX_SCALAR: u32 = 0x10_FFFF;

/// Appends the byte-range sequences that encode `lo..=hi`, surrogates
/// excluded.
pub(crate) fn scalar_range_sequences(lo: u32, hi: u32, out: &mut Vec<ByteRanges>) {
    let hi = hi.min(MAX_SCALAR);
    if lo > hi {
        return;
    }
    let mut pending = Vec::new();
    if lo < SURROGATES.0 {
        pending.push((lo, hi.min(SURROGATES.0 - 1)));
    }
    if hi > SURROGATES.1 {
        pending.push((lo.max(SURROGATES.1 + 1), hi));
    }
    while let Some((lo, hi)) = pending.pop() {
        if lo > hi {
            continue;
        }
        // Split where the encoded length changes.
        if let Some(limit) = [0x7F, 0x7FF, 0xFFFF]
            .into_iter()
            .find(|&limit| lo <= limit && hi > limit)
        {
            pending.push((lo, limit));
            pending.push((limit + 1, hi));
            continue;
        }
        if hi <= 0x7F {
            out.push(vec![(lo as u8, hi as u8)]);
            continue;
        }
        // Split until the continuation bytes of lo and hi span full ranges
        // below the first byte that differs.
        let length = encoded_length(lo);
        let mut split = None;
        for index in 1..length {
            let mask = (1_u32 << (6 * index)) - 1;
            if lo & !mask != hi & !mask {
                if lo & mask != 0 {
                    split = Some(((lo, lo | mask), ((lo | mask) + 1, hi)));
                    break;
                }
                if hi & mask != mask {
                    split = Some(((lo, (hi & !mask) - 1), (hi & !mask, hi)));
                    break;
                }
            }
        }
        if let Some((first, second)) = split {
            pending.push(second);
            pending.push(first);
            continue;
        }
        let low = encode(lo);
        let high = encode(hi);
        out.push(
            low.iter()
                .zip(high.iter())
                .take(length)
                .map(|(&low, &high)| (low, high))
                .collect(),
        );
    }
}

fn encoded_length(scalar: u32) -> usize {
    match scalar {
        0..=0x7F => 1,
        0x80..=0x7FF => 2,
        0x800..=0xFFFF => 3,
        _ => 4,
    }
}

fn encode(scalar: u32) -> [u8; 4] {
    let mut bytes = [0; 4];
    char::from_u32(scalar)
        .expect("surrogates were removed")
        .encode_utf8(&mut bytes);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(sequences: &[ByteRanges], bytes: &[u8]) -> bool {
        sequences.iter().any(|ranges| {
            ranges.len() == bytes.len()
                && ranges
                    .iter()
                    .zip(bytes)
                    .all(|(&(lo, hi), byte)| (lo..=hi).contains(byte))
        })
    }

    #[test]
    fn sequences_encode_exactly_the_range() {
        let ranges = [
            (0, 0x10_FFFF),
            (0x20, 0x7E),
            (0x7F, 0x80),
            (0x7FF, 0x800),
            (0xD7FF, 0xE000),
            (0x1_0000, 0x1_0FFF),
            (0x3A5, 0x2_1234),
            (0xFFFF, 0x1_0000),
        ];
        for (lo, hi) in ranges {
            let mut sequences = Vec::new();
            scalar_range_sequences(lo, hi, &mut sequences);
            for scalar in (0..=0x10_FFFF)
                .step_by(97)
                .chain([lo, hi, 0x7F, 0x80, 0xFFFF])
            {
                let Some(character) = char::from_u32(scalar) else {
                    continue;
                };
                let mut buffer = [0; 4];
                let encoded = character.encode_utf8(&mut buffer).as_bytes();
                assert_eq!(
                    matches(&sequences, encoded),
                    (lo..=hi).contains(&scalar),
                    "{scalar:#x} in {lo:#x}..={hi:#x}"
                );
            }
            // Surrogate encodings and overlong forms never match.
            assert!(!matches(&sequences, &[0xED, 0xA0, 0x80]));
            assert!(!matches(&sequences, &[0xC0, 0x80]));
            assert!(!matches(&sequences, &[0xE0, 0x80, 0x80]));
        }
    }
}
