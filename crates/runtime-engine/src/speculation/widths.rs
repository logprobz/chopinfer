//! Verify widths a package runs at its prepared speed, for balanced plans to
//! land on.
//!
//! A package may prepare its fastest plans for some verify row counts only,
//! so a step between them runs slower than the next prepared one up:
//! A four-rank serving check (300 W, 2026-10-23) verified 24 rows of four items
//! in 43.4 ms and 32 rows of eight in 37.6 ms. A balanced plan with widths
//! verifies the widest one between its decode rows and the rows its depth
//! limits reach, filled by acceptance. Like the rest of a balanced plan it
//! depends on the batch alone, never on measured step times.

/// Widths one list may hold.
pub const MAX_VERIFY_WIDTHS: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerifyWidthsError {
    Empty,
    TooMany {
        maximum: usize,
    },
    /// Widths must be positive and strictly increasing.
    Invalid,
    Syntax,
}

/// Verify row counts by increasing size. The default list is empty and
/// leaves every plan at the rows its limits reach.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct VerifyWidths {
    widths: [u16; MAX_VERIFY_WIDTHS],
    len: u8,
}

impl std::fmt::Debug for VerifyWidths {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_list().entries(self.widths()).finish()
    }
}

impl VerifyWidths {
    /// The empty list.
    pub const NONE: Self = Self {
        widths: [0; MAX_VERIFY_WIDTHS],
        len: 0,
    };

    pub fn new(widths: &[u16]) -> Result<Self, VerifyWidthsError> {
        if widths.is_empty() {
            return Err(VerifyWidthsError::Empty);
        }
        if widths.len() > MAX_VERIFY_WIDTHS {
            return Err(VerifyWidthsError::TooMany {
                maximum: MAX_VERIFY_WIDTHS,
            });
        }
        if widths.contains(&0) || widths.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(VerifyWidthsError::Invalid);
        }
        let mut list = Self::default();
        list.widths[..widths.len()].copy_from_slice(widths);
        list.len = widths.len() as u8;
        Ok(list)
    }

    /// Parses widths separated by commas, for example `6,8,16,32`.
    pub fn parse(text: &str) -> Result<Self, VerifyWidthsError> {
        let widths = text
            .split(',')
            .map(|width| {
                width
                    .trim()
                    .parse::<u16>()
                    .map_err(|_| VerifyWidthsError::Syntax)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(&widths)
    }

    pub fn widths(&self) -> &[u16] {
        &self.widths[..usize::from(self.len)]
    }

    /// The widest width from `low` through `high` rows, if any.
    pub fn widest_within(&self, low: usize, high: usize) -> Option<usize> {
        self.widths()
            .iter()
            .rev()
            .map(|&width| usize::from(width))
            .find(|width| (low..=high).contains(width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_widest_width_within_the_rows_is_taken() {
        let widths = VerifyWidths::parse("6, 8,16,32").unwrap();
        assert_eq!(widths.widths(), [6, 8, 16, 32]);
        for (low, high, widest) in [
            (1, 6, Some(6)),
            (2, 12, Some(8)),
            (4, 24, Some(16)),
            (5, 30, Some(16)),
            (8, 32, Some(32)),
            (9, 15, None),
            (17, 31, None),
            (1, 5, None),
        ] {
            assert_eq!(widths.widest_within(low, high), widest, "{low}..={high}");
        }
        assert_eq!(VerifyWidths::default().widest_within(1, 32), None);
    }

    #[test]
    fn width_lists_refuse_bad_entries() {
        assert_eq!(VerifyWidths::parse(""), Err(VerifyWidthsError::Syntax));
        assert_eq!(VerifyWidths::parse("8,x"), Err(VerifyWidthsError::Syntax));
        assert_eq!(VerifyWidths::new(&[]), Err(VerifyWidthsError::Empty));
        for text in ["0,8", "8,8", "16,8"] {
            assert_eq!(
                VerifyWidths::parse(text),
                Err(VerifyWidthsError::Invalid),
                "{text}"
            );
        }
        assert_eq!(
            VerifyWidths::parse("1,2,3,4,5,6,7,8,9"),
            Err(VerifyWidthsError::TooMany {
                maximum: MAX_VERIFY_WIDTHS
            })
        );
    }
}
