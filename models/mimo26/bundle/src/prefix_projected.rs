//! Exact DFlash derived state appended after the target ring and raw history.
//! Copies retain BF16 bits and absolute-position circular addressing. The
//! cursor is data in the snapshot header, so relocation needs no
//! source-request map.

use bundle_runtime::prefix::{HeaderLayout, Word};

/// The DFlash snapshot header, after the projected context on every rank:
/// the boundary, the projection cursor and the rank count.
pub(crate) const HEADER: HeaderLayout = HeaderLayout {
    magic: *b"DFCTX001",
    words: &[
        Word::Tokens,
        Word::Cursor,
        Word::Ranks,
        Word::Zero,
        Word::Zero,
        Word::Zero,
        Word::Zero,
    ],
};
#[cfg(test)]
pub(crate) const HEADER_BYTES: u64 = HEADER.bytes();

#[cfg(test)]
mod tests {
    use bundle_runtime::prefix::{Header, Stamp};

    use super::*;

    /// The DFlash header keeps the bytes of its first layout: the magic, the
    /// boundary, the cursor or `u64::MAX`, the rank count and four zero
    /// words.
    #[test]
    fn the_dflash_header_keeps_its_bytes() {
        let stamp = Stamp {
            ranks: 4,
            digest: 0,
        };
        for cursor in [None, Some(0), Some(1020), Some(1024)] {
            let header = Header {
                tokens: 1024,
                frontier: None,
                owner: None,
                cursor,
            };
            let bytes = header.encode(&HEADER, stamp).unwrap();
            let words = [
                u64::from_le_bytes(*b"DFCTX001"),
                1024,
                cursor.unwrap_or(u64::MAX),
                4,
                0,
                0,
                0,
                0,
            ];
            assert_eq!(
                bytes,
                words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>()
            );
            assert_eq!(Header::decode(&bytes, &HEADER, stamp, 1024), Ok(header));
            assert!(Header::decode(&bytes, &HEADER, stamp, 1025).is_err());
            for index in [0, 24, 32, 63] {
                let mut corrupt = bytes.clone();
                corrupt[index] ^= 1;
                assert!(Header::decode(&corrupt, &HEADER, stamp, 1024).is_err());
            }
        }
        let past = Header {
            tokens: 1024,
            frontier: None,
            owner: None,
            cursor: Some(1025),
        };
        assert!(past.encode(&HEADER, stamp).is_err());
    }
}
