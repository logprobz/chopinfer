//! Block sets: `even`, `odd`, or ascending, separated indices and ranges
//! such as `1-8,10-17`. Each set has exactly one spelling, so equal sets
//! give equal digests.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Debug, Eq, PartialEq)]
enum Spec {
    Even,
    Odd,
    Ranges(Vec<(u32, u32)>),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockSet {
    text: String,
    spec: Spec,
}

impl BlockSet {
    pub fn parse(text: &str) -> Result<Self, String> {
        let malformed = || {
            format!(
                "block set `{text}` is not `even`, `odd` or ascending, separated `a` and `a-b` items"
            )
        };
        let spec = match text {
            "even" => Spec::Even,
            "odd" => Spec::Odd,
            _ => {
                let mut ranges: Vec<(u32, u32)> = Vec::new();
                for item in text.split(',') {
                    let number = |digits: &str| {
                        (!digits.is_empty()
                            && (digits == "0" || !digits.starts_with('0'))
                            && digits.bytes().all(|b| b.is_ascii_digit()))
                        .then(|| digits.parse::<u32>().ok())
                        .flatten()
                        .ok_or_else(malformed)
                    };
                    let range = match item.split_once('-') {
                        Some((first, last)) => (number(first)?, number(last)?),
                        None => (number(item)?, number(item)?),
                    };
                    let after_previous = ranges
                        .last()
                        .is_none_or(|previous| u64::from(range.0) > u64::from(previous.1) + 1);
                    let ordered = range.0 < range.1 || !item.contains('-') && range.0 == range.1;
                    if !after_previous || !ordered {
                        return Err(malformed());
                    }
                    ranges.push(range);
                }
                Spec::Ranges(ranges)
            }
        };
        Ok(Self {
            text: text.to_owned(),
            spec,
        })
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn contains(&self, block: u32) -> bool {
        match &self.spec {
            Spec::Even => block % 2 == 0,
            Spec::Odd => block % 2 == 1,
            Spec::Ranges(ranges) => ranges
                .iter()
                .any(|&(first, last)| (first..=last).contains(&block)),
        }
    }

    /// The members below `blocks`, refusing a set that names a block past it
    /// or names none.
    pub fn members(&self, blocks: u32) -> Result<Vec<u32>, String> {
        if let Spec::Ranges(ranges) = &self.spec
            && ranges.last().is_some_and(|&(_, last)| last >= blocks)
        {
            return Err(format!(
                "block set `{}` names a block past the last of {blocks}",
                self.text
            ));
        }
        let members: Vec<u32> = (0..blocks).filter(|&block| self.contains(block)).collect();
        if members.is_empty() {
            return Err(format!("block set `{}` names no block", self.text));
        }
        Ok(members)
    }

    /// The members below `blocks`, refusing a set with a shorter spelling
    /// there: every block (state it without a set) or the even or odd blocks
    /// spelled as a list.
    pub fn canonical_members(&self, blocks: u32) -> Result<Vec<u32>, String> {
        let members = self.members(blocks)?;
        if members.len() == blocks as usize {
            return Err(format!(
                "block set `{}` names every block of {blocks}; state it without a set",
                self.text
            ));
        }
        if let Spec::Ranges(_) = self.spec {
            for (parity, name) in [(0, "even"), (1, "odd")] {
                if members == (0..blocks).filter(|b| b % 2 == parity).collect::<Vec<_>>() {
                    return Err(format!(
                        "block set `{}` is `{name}`; write it so",
                        self.text
                    ));
                }
            }
        }
        Ok(members)
    }
}

impl Serialize for BlockSet {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.text)
    }
}

impl<'de> Deserialize<'de> for BlockSet {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        Self::parse(&text).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_sets_have_one_spelling() {
        let set = BlockSet::parse("1-8,10-17,19-26").unwrap();
        assert_eq!(set.members(28).unwrap().len(), 24);
        assert!(!set.contains(9));
        assert_eq!(
            BlockSet::parse("even").unwrap().members(24).unwrap().len(),
            12
        );
        assert_eq!(
            BlockSet::parse("0,9,18,27").unwrap().members(28).unwrap(),
            [0, 9, 18, 27]
        );
        assert!(BlockSet::parse("0,9").unwrap().members(9).is_err());
        for bad in [
            "", "1-1", "2-1", "1,1", "1,2", "1-3,4", "3,1", "01", "1 ,2", "all",
        ] {
            assert!(BlockSet::parse(bad).is_err(), "accepted `{bad}`");
        }
    }
}
