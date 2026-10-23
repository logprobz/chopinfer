//! Most draft positions a balanced plan gives each item, by concurrency.
//!
//! A schedule fixes where drafting pays from served measurements: a batch of
//! `items` items drafts at most the depth of the last entry at or below
//! `items`, and depth 0 serves those concurrencies without drafts. Like the
//! rest of a balanced plan it depends on the batch alone, never on measured
//! step times.

use super::MAX_DRAFT_TOKENS;

/// Entries one schedule may hold.
pub const MAX_DRAFT_DEPTHS: usize = 8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DraftDepthsError {
    Empty,
    TooMany {
        maximum: usize,
    },
    /// Items must be positive and strictly increasing, depths at most
    /// `MAX_DRAFT_TOKENS`.
    Invalid,
    Syntax,
}

/// A schedule of `items:depth` entries by increasing items. The default
/// schedule is empty and caps nothing.
#[derive(Clone, Copy, Default, Eq, PartialEq)]
pub struct DraftDepths {
    entries: [(u16, u8); MAX_DRAFT_DEPTHS],
    len: u8,
}

impl std::fmt::Debug for DraftDepths {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_list().entries(self.entries()).finish()
    }
}

impl DraftDepths {
    pub fn new(entries: &[(u16, u8)]) -> Result<Self, DraftDepthsError> {
        if entries.is_empty() {
            return Err(DraftDepthsError::Empty);
        }
        if entries.len() > MAX_DRAFT_DEPTHS {
            return Err(DraftDepthsError::TooMany {
                maximum: MAX_DRAFT_DEPTHS,
            });
        }
        if entries
            .iter()
            .any(|&(items, depth)| items == 0 || usize::from(depth) > MAX_DRAFT_TOKENS)
            || entries.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        {
            return Err(DraftDepthsError::Invalid);
        }
        let mut schedule = Self::default();
        schedule.entries[..entries.len()].copy_from_slice(entries);
        schedule.len = entries.len() as u8;
        Ok(schedule)
    }

    /// Parses `items:depth` entries separated by commas, for example
    /// `1:5,8:3,16:0`.
    pub fn parse(text: &str) -> Result<Self, DraftDepthsError> {
        let entries = text
            .split(',')
            .map(|entry| {
                let (items, depth) = entry.split_once(':').ok_or(DraftDepthsError::Syntax)?;
                let items = items
                    .trim()
                    .parse::<u16>()
                    .map_err(|_| DraftDepthsError::Syntax)?;
                let depth = depth
                    .trim()
                    .parse::<u8>()
                    .map_err(|_| DraftDepthsError::Syntax)?;
                Ok((items, depth))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(&entries)
    }

    pub fn entries(&self) -> &[(u16, u8)] {
        &self.entries[..usize::from(self.len)]
    }

    /// The most drafts one item of a batch of `items` items may verify:
    /// the depth of the last entry at or below `items`, or `None` below
    /// every entry and for the empty schedule.
    pub fn at(&self, items: usize) -> Option<usize> {
        self.entries()
            .iter()
            .take_while(|&&(from, _)| usize::from(from) <= items)
            .last()
            .map(|&(_, depth)| usize::from(depth))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_schedule_holds_from_each_entry_to_the_next() {
        let depths = DraftDepths::parse("1:5, 8:3,16:0").unwrap();
        assert_eq!(depths.entries(), [(1, 5), (8, 3), (16, 0)]);
        for (items, depth) in [(1, 5), (7, 5), (8, 3), (15, 3), (16, 0), (32, 0)] {
            assert_eq!(depths.at(items), Some(depth), "{items}");
        }
        let later = DraftDepths::parse("8:2").unwrap();
        assert_eq!(later.at(4), None);
        assert_eq!(later.at(8), Some(2));
        assert_eq!(DraftDepths::default().at(1), None);
    }

    #[test]
    fn schedules_refuse_bad_entries() {
        assert_eq!(DraftDepths::parse(""), Err(DraftDepthsError::Syntax));
        assert_eq!(DraftDepths::parse("8"), Err(DraftDepthsError::Syntax));
        assert_eq!(DraftDepths::parse("8:x"), Err(DraftDepthsError::Syntax));
        assert_eq!(DraftDepths::new(&[]), Err(DraftDepthsError::Empty));
        for text in ["0:3", "8:3,8:2", "16:0,8:3", "1:17"] {
            assert_eq!(
                DraftDepths::parse(text),
                Err(DraftDepthsError::Invalid),
                "{text}"
            );
        }
        let many = (1..=9).map(|n| format!("{n}:1")).collect::<Vec<_>>();
        assert_eq!(
            DraftDepths::parse(&many.join(",")),
            Err(DraftDepthsError::TooMany {
                maximum: MAX_DRAFT_DEPTHS
            })
        );
    }
}
