//! Which marker spellings in a completion are markers, when the host sends
//! the byte ranges of the tokens generated as the declaration's ID-only
//! markers (`vocabulary.id_only_markers`).
//!
//! Text without token boundaries, or a parser whose pushes carry no marks,
//! reads every marker by its spelling. With marks, a marker is read as the
//! tokens it is made of: its ID-only spellings, found leftmost and longest
//! first, each one marked token, and every other byte of it from ordinary
//! tokens. An occurrence is the marker only when each ID-only spelling in it
//! is exactly one mark and no other byte of it lies in a mark, so the same
//! spelling generated as ordinary tokens is text.

use std::ops::Range;

/// What a completion's marks let through as markers.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MarkMatchV1<'a> {
    /// The ID-only spellings and the completion's marks, ascending; `None`
    /// to read every marker by its spelling.
    marked: Option<(&'a [String], &'a [Range<usize>])>,
}

impl<'a> MarkMatchV1<'a> {
    /// Every marker read by its spelling.
    pub(crate) const SPELLED: Self = Self { marked: None };

    /// The markers `marks` let through, for the ID-only `spellings`.
    pub(crate) const fn marked(spellings: &'a [String], marks: &'a [Range<usize>]) -> Self {
        Self {
            marked: Some((spellings, marks)),
        }
    }

    /// Whether every marker is read by its spelling.
    pub(crate) const fn spelled(&self) -> bool {
        self.marked.is_none()
    }

    /// Whether `marker`, spelled at completion byte `at`, is the marker.
    pub(crate) fn accepts(&self, at: usize, marker: &str) -> bool {
        self.admits(at, marker, marker.len())
    }

    /// Whether the first `present` bytes of `marker`, spelled at completion
    /// byte `at` and ending where the released bytes end, can still be the
    /// marker once later bytes complete it: each ID-only spelling they hold
    /// whole is exactly one mark, and no other byte of them lies in a mark.
    /// A spelling they cut short is a token not yet whole, which has no mark
    /// until its last byte is released.
    pub(crate) fn admits(&self, at: usize, marker: &str, present: usize) -> bool {
        let Some((spellings, marks)) = self.marked else {
            return true;
        };
        let end = at + present;
        let first = marks.partition_point(|mark| mark.end <= at);
        let mut marks = marks[first..].iter().take_while(|mark| mark.start < end);
        for part in spelling_parts_v1(marker, spellings) {
            if part.end > present {
                break;
            }
            if marks.next() != Some(&(at + part.start..at + part.end)) {
                return false;
            }
        }
        marks.next().is_none()
    }
}

/// `pieces` joined, with a mark on each ID-only spelling in a piece flagged
/// as markup, as a tool-choice grammar writes those spellings as tokens and
/// everything else as ordinary tokens.
pub(crate) fn marked_join_v1(
    spellings: &[String],
    pieces: &[(&str, bool)],
) -> (String, Vec<Range<usize>>) {
    let mut text = String::new();
    let mut marks = Vec::new();
    for &(piece, markup) in pieces {
        if markup {
            let at = text.len();
            marks.extend(
                spelling_parts_v1(piece, spellings).map(|part| at + part.start..at + part.end),
            );
        }
        text.push_str(piece);
    }
    (text, marks)
}

/// The byte ranges of `marker` its ID-only spellings take, leftmost and
/// longest first.
pub(crate) fn spelling_parts_v1<'a>(
    marker: &'a str,
    spellings: &'a [String],
) -> impl Iterator<Item = Range<usize>> + 'a {
    let marker = marker.as_bytes();
    let mut at = 0;
    std::iter::from_fn(move || {
        while at < marker.len() {
            let found = spellings
                .iter()
                .map(String::as_bytes)
                .filter(|spelling| !spelling.is_empty() && marker[at..].starts_with(spelling))
                .map(<[u8]>::len)
                .max();
            if let Some(length) = found {
                let part = at..at + length;
                at += length;
                return Some(part);
            }
            at += 1;
        }
        None
    })
}

/// Where the character boundaries of a replacement decoding of some bytes
/// lie in those bytes, so a marker found in a tool region's decoded text is
/// checked against the marks of the bytes it came from.
#[derive(Debug)]
pub(crate) struct SourceMapV1 {
    /// Each run of valid bytes: where it begins in the decoded text and in
    /// the bytes, and its length. An invalid sequence after a run decodes
    /// as one replacement character.
    runs: Vec<(usize, usize, usize)>,
    /// The length of the bytes.
    length: usize,
}

impl SourceMapV1 {
    pub(crate) fn new(bytes: &[u8]) -> Self {
        let (mut runs, mut decoded, mut source) = (Vec::new(), 0, 0);
        for chunk in bytes.utf8_chunks() {
            runs.push((decoded, source, chunk.valid().len()));
            decoded += chunk.valid().len();
            source += chunk.valid().len() + chunk.invalid().len();
            if !chunk.invalid().is_empty() {
                decoded += '\u{fffd}'.len_utf8();
            }
        }
        Self {
            runs,
            length: bytes.len(),
        }
    }

    /// The byte offset of the decoded text's character boundary `decoded`.
    pub(crate) fn source(&self, decoded: usize) -> usize {
        let index = self.runs.partition_point(|run| run.0 <= decoded);
        match index.checked_sub(1).map(|run| self.runs[run]) {
            Some((text, source, valid)) if decoded - text <= valid => source + decoded - text,
            // Past a run's replacement character: where the next run begins.
            _ => self.runs.get(index).map_or(self.length, |run| run.1),
        }
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    fn spellings() -> Vec<String> {
        vec!["|T|".into(), "</t>".into()]
    }

    #[test]
    fn a_marker_is_its_marked_spellings_and_unmarked_bytes() {
        let spellings = spellings();
        let marker = "<|T| calls>";
        // The marker at byte 10: its `|T|` at 11..14.
        let marks = [2..5, 11..14, 30..33];
        let check = MarkMatchV1::marked(&spellings, &marks);
        assert!(check.accepts(10, marker));
        assert!(!check.accepts(20, marker));
        // A mark over a plain byte, or one that is not the spelling whole.
        for marks in [
            vec![10..14],
            vec![11..14, 15..16],
            vec![11..13],
            vec![12..14],
            vec![],
        ] {
            assert!(
                !MarkMatchV1::marked(&spellings, &marks).accepts(10, marker),
                "{marks:?}"
            );
        }
        // A marker with no ID-only spelling takes no mark.
        assert!(check.accepts(5, "\">\n"));
        assert!(!check.accepts(4, "\">\n"));
        assert!(MarkMatchV1::SPELLED.accepts(4, "\">\n"));
        assert!(MarkMatchV1::marked(&spellings, &[]).accepts(0, "abc"));
        // Spellings are taken leftmost and longest first.
        let marks = [0..3, 3..7];
        assert!(MarkMatchV1::marked(&spellings, &marks).accepts(0, "|T|</t>"));
        assert!(!MarkMatchV1::marked(&spellings, &[0..7]).accepts(0, "|T|</t>"));
    }

    #[test]
    fn a_prefix_holds_only_what_can_still_be_the_marker() {
        let spellings = spellings();
        let marker = "<|T| calls>";
        let marked = MarkMatchV1::marked(&spellings, &[1..4]);
        // `<|T| c` with the spelling marked, and the bare `<`.
        assert!(marked.admits(0, marker, 6));
        assert!(marked.admits(0, marker, 1));
        // A spelling cut short is a token not yet whole, so it holds only
        // without a mark.
        assert!(MarkMatchV1::marked(&spellings, &[]).admits(0, marker, 3));
        assert!(!MarkMatchV1::marked(&spellings, &[1..3]).admits(0, marker, 3));
        assert!(!MarkMatchV1::marked(&spellings, &[2..3]).admits(0, marker, 3));
        // The spelling whole but from ordinary tokens, or a plain byte marked.
        assert!(!MarkMatchV1::marked(&spellings, &[]).admits(0, marker, 6));
        assert!(!MarkMatchV1::marked(&spellings, &[0..1]).admits(0, marker, 1));
    }

    #[test]
    fn a_source_map_finds_the_bytes_of_each_character_boundary() {
        for bytes in [
            &b"plain"[..],
            b"a\xffb",
            b"\xff\xfe",
            b"ab\xe2\x82",
            b"\xe2\x82ab\xffc",
        ] {
            let map = SourceMapV1::new(bytes);
            let text = String::from_utf8_lossy(bytes);
            let mut expected = Vec::new();
            let (mut decoded, mut source) = (0, 0);
            for chunk in bytes.utf8_chunks() {
                for (offset, _) in chunk.valid().char_indices() {
                    expected.push((decoded + offset, source + offset));
                }
                decoded += chunk.valid().len();
                source += chunk.valid().len();
                if !chunk.invalid().is_empty() {
                    expected.push((decoded, source));
                    decoded += '\u{fffd}'.len_utf8();
                    source += chunk.invalid().len();
                }
            }
            expected.push((text.len(), bytes.len()));
            for (decoded, source) in expected {
                assert!(text.is_char_boundary(decoded));
                assert_eq!(map.source(decoded), source, "{bytes:?} at {decoded}");
            }
        }
    }
}
