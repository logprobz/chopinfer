//! The grammar every constraint compiles through.
//!
//! A grammar is a list of rules. Each rule is an expression over UTF-8 text
//! that may refer to other rules, which is how recursive structures such as
//! nested JSON values are written. Constraint front ends (JSON Schema, regular
//! expressions, choices, and the tool-call builders frontends write) all
//! produce this form.

use std::fmt;

/// The largest Unicode scalar value.
pub const MAX_SCALAR: u32 = 0x10_FFFF;

/// A set of Unicode scalar values, kept as sorted disjoint ranges.
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct CharClass {
    ranges: Vec<(u32, u32)>,
}

impl CharClass {
    /// The class of every scalar in the given inclusive ranges.
    #[must_use]
    pub fn from_ranges(ranges: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let mut ranges = ranges
            .into_iter()
            .filter(|&(lo, hi)| lo <= hi && lo <= MAX_SCALAR)
            .map(|(lo, hi)| (lo, hi.min(MAX_SCALAR)))
            .collect::<Vec<_>>();
        ranges.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
        for (lo, hi) in ranges {
            match merged.last_mut() {
                Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
                _ => merged.push((lo, hi)),
            }
        }
        Self { ranges: merged }
    }

    /// A class of the given characters.
    #[must_use]
    pub fn of(characters: &str) -> Self {
        Self::from_ranges(characters.chars().map(|c| (c as u32, c as u32)))
    }

    /// Every scalar value.
    #[must_use]
    pub fn any() -> Self {
        Self::from_ranges([(0, MAX_SCALAR)])
    }

    #[must_use]
    pub fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    #[must_use]
    pub fn contains(&self, scalar: u32) -> bool {
        let index = self.ranges.partition_point(|&(_, hi)| hi < scalar);
        self.ranges
            .get(index)
            .is_some_and(|&(lo, hi)| lo <= scalar && scalar <= hi)
    }

    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        Self::from_ranges(self.ranges.iter().chain(&other.ranges).copied())
    }

    #[must_use]
    pub fn complement(&self) -> Self {
        let mut ranges = Vec::with_capacity(self.ranges.len() + 1);
        let mut next = 0_u32;
        for &(lo, hi) in &self.ranges {
            if lo > next {
                ranges.push((next, lo - 1));
            }
            next = hi.saturating_add(1);
        }
        if next <= MAX_SCALAR {
            ranges.push((next, MAX_SCALAR));
        }
        Self { ranges }
    }

    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        self.complement().union(&other.complement()).complement()
    }
}

/// Index of one rule in its grammar.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuleId(pub(crate) u32);

/// A regular expression over text, extended with references to rules.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum Expr {
    /// The empty string.
    Empty,
    /// No string at all.
    Nothing,
    /// Exactly these bytes.
    Literal(Vec<u8>),
    /// One scalar value from the class, UTF-8 encoded.
    Class(CharClass),
    Seq(Vec<Expr>),
    Alt(Vec<Expr>),
    /// Between `min` and `max` repetitions; no `max` means unbounded.
    Repeat {
        expr: Box<Expr>,
        min: u32,
        max: Option<u32>,
    },
    /// Whatever the referenced rule matches.
    Rule(RuleId),
    /// Any UTF-8 text that contains none of the markers.
    TextExcluding(Vec<Vec<u8>>),
    /// UTF-8 text through the first occurrence of `end`, which ends it,
    /// containing none of `avoid`. Occurrences that straddle the text and
    /// `end` count, so a reader that splits at the first `end` agrees.
    TextUntil {
        end: Vec<u8>,
        avoid: Vec<Vec<u8>>,
    },
    /// A subsequence of `items` in their order that includes every required
    /// item, then any number of `tail` items, with `separator` between
    /// consecutive entries. This is how objects list their properties in
    /// schema order without writing each optional combination out.
    OrderedSubset {
        items: Vec<(Expr, bool)>,
        tail: Option<Box<Expr>>,
        separator: Box<Expr>,
    },
}

impl Expr {
    #[must_use]
    pub fn literal(text: &str) -> Self {
        Self::Literal(text.as_bytes().to_vec())
    }

    #[must_use]
    pub fn seq(items: impl IntoIterator<Item = Expr>) -> Self {
        let mut flat = Vec::new();
        for item in items {
            match item {
                Self::Empty => {}
                Self::Seq(inner) => flat.extend(inner),
                Self::Nothing => return Self::Nothing,
                item => flat.push(item),
            }
        }
        match flat.len() {
            0 => Self::Empty,
            1 => flat.pop().expect("one item"),
            _ => Self::Seq(flat),
        }
    }

    #[must_use]
    pub fn alt(items: impl IntoIterator<Item = Expr>) -> Self {
        let mut flat = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for item in items {
            match item {
                Self::Nothing => {}
                Self::Alt(inner) => {
                    // Preserve flattened branch order and duplicates as before.
                    seen.extend(inner.iter().cloned());
                    flat.extend(inner);
                }
                item => {
                    if seen.insert(item.clone()) {
                        flat.push(item);
                    }
                }
            }
        }
        match flat.len() {
            0 => Self::Nothing,
            1 => flat.pop().expect("one item"),
            _ => Self::Alt(flat),
        }
    }

    #[must_use]
    pub fn repeat(self, min: u32, max: Option<u32>) -> Self {
        if max == Some(0) {
            return Self::Empty;
        }
        if min == 1 && max == Some(1) {
            return self;
        }
        Self::Repeat {
            expr: Box::new(self),
            min,
            max,
        }
    }

    #[must_use]
    pub fn optional(self) -> Self {
        self.repeat(0, Some(1))
    }

    #[must_use]
    pub fn star(self) -> Self {
        self.repeat(0, None)
    }

    #[must_use]
    pub fn plus(self) -> Self {
        self.repeat(1, None)
    }
}

/// A complete grammar: its rules and the one that matches whole outputs.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Grammar {
    pub(crate) rules: Vec<Expr>,
    pub(crate) root: RuleId,
    pub(crate) special_tokens: Vec<Vec<u8>>,
}

impl Grammar {
    /// A grammar of one expression without rules.
    #[must_use]
    pub fn from_expr(expr: Expr) -> Self {
        Self {
            rules: vec![expr],
            root: RuleId(0),
            special_tokens: Vec::new(),
        }
    }

    #[must_use]
    pub fn root(&self) -> RuleId {
        self.root
    }

    #[must_use]
    pub fn rule(&self, id: RuleId) -> &Expr {
        &self.rules[id.0 as usize]
    }

    /// Heap bytes of this grammar held in an `Arc`, an upper estimate from
    /// the capacities of its rules and special token texts.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        use crate::heap::{arc_bytes, vec_bytes};
        let texts = |texts: &Vec<Vec<u8>>| {
            vec_bytes::<Vec<u8>>(texts.capacity()) + texts.iter().map(Vec::capacity).sum::<usize>()
        };
        let mut bytes = arc_bytes::<Self>()
            + vec_bytes::<Expr>(self.rules.capacity())
            + texts(&self.special_tokens);
        let mut pending = self.rules.iter().collect::<Vec<_>>();
        while let Some(expr) = pending.pop() {
            bytes += match expr {
                Expr::Empty | Expr::Nothing | Expr::Rule(_) => 0,
                Expr::Literal(bytes) => bytes.capacity(),
                Expr::Class(class) => vec_bytes::<(u32, u32)>(class.ranges.capacity()),
                Expr::Seq(items) | Expr::Alt(items) => {
                    pending.extend(items);
                    vec_bytes::<Expr>(items.capacity())
                }
                Expr::Repeat { expr, .. } => {
                    pending.push(expr);
                    size_of::<Expr>()
                }
                Expr::TextExcluding(markers) => texts(markers),
                Expr::TextUntil { end, avoid } => end.capacity() + texts(avoid),
                Expr::OrderedSubset {
                    items,
                    tail,
                    separator,
                } => {
                    pending.extend(items.iter().map(|(item, _)| item));
                    pending.push(separator);
                    if let Some(tail) = tail {
                        pending.push(tail);
                    }
                    vec_bytes::<(Expr, bool)>(items.capacity())
                        + size_of::<Expr>() * (1 + usize::from(tail.is_some()))
                }
            };
        }
        bytes
    }

    /// The text of special tokens this grammar admits wherever their bytes
    /// fit; other special tokens never match.
    #[must_use]
    pub fn special_tokens(&self) -> &[Vec<u8>] {
        &self.special_tokens
    }

    /// Also admits the special tokens whose text is one of these.
    #[must_use]
    pub fn with_special_tokens(mut self, texts: impl IntoIterator<Item = Vec<u8>>) -> Self {
        for text in texts {
            if !self.special_tokens.contains(&text) {
                self.special_tokens.push(text);
            }
        }
        self
    }
}

/// Builds a grammar whose rules may refer to each other.
#[derive(Debug, Default)]
pub struct GrammarBuilder {
    rules: Vec<Option<Expr>>,
}

impl GrammarBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Reserves a rule that is defined later, so it can be referenced first.
    pub fn declare(&mut self) -> RuleId {
        self.rules.push(None);
        RuleId(u32::try_from(self.rules.len() - 1).expect("rule count fits u32"))
    }

    pub fn define(&mut self, id: RuleId, expr: Expr) {
        self.rules[id.0 as usize] = Some(expr);
    }

    /// Adds a rule and returns a reference to it.
    pub fn rule(&mut self, expr: Expr) -> RuleId {
        let id = self.declare();
        self.define(id, expr);
        id
    }

    pub fn finish(self, root: RuleId) -> Result<Grammar, GrammarError> {
        let rules = self
            .rules
            .into_iter()
            .map(|rule| rule.ok_or_else(|| GrammarError::invalid("a grammar rule is undefined")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Grammar {
            rules,
            root,
            special_tokens: Vec::new(),
        })
    }
}

/// Why a constraint could not be built, compiled, or followed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GrammarError {
    /// The constraint uses a feature outside the supported subset.
    Unsupported(String),
    /// The constraint itself is malformed.
    Invalid(String),
    /// The constraint compiles past a resource bound.
    TooLarge(String),
    /// A token was offered that the constraint does not admit.
    Rejected { token_id: u32 },
    /// No token in the vocabulary continues the constraint.
    NoAllowedToken,
}

impl GrammarError {
    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    pub(crate) fn too_large(message: impl Into<String>) -> Self {
        Self::TooLarge(message.into())
    }
}

impl fmt::Display for GrammarError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(message) | Self::Invalid(message) | Self::TooLarge(message) => {
                formatter.write_str(message)
            }
            Self::Rejected { token_id } => {
                write!(
                    formatter,
                    "token {token_id} does not continue the constraint"
                )
            }
            Self::NoAllowedToken => formatter.write_str("no token continues the constraint"),
        }
    }
}

impl std::error::Error for GrammarError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_merge_and_complement() {
        let class = CharClass::from_ranges([(5, 9), (0, 3), (4, 4), (20, 30)]);
        assert_eq!(class.ranges(), &[(0, 9), (20, 30)]);
        assert!(class.contains(9) && !class.contains(10) && class.contains(25));
        let complement = class.complement();
        assert_eq!(complement.ranges(), &[(10, 19), (31, MAX_SCALAR)]);
        assert_eq!(complement.complement(), class);
        let other = CharClass::from_ranges([(8, 22)]);
        assert_eq!(class.intersection(&other).ranges(), &[(8, 9), (20, 22)]);
    }
}
