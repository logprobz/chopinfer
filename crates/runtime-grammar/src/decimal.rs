//! Finite JSON numbers between integer bounds, compared by value
//! ([`crate::SchemaCompiler::finite_numbers`]).
//!
//! A number without an exponent is a [`number_range`] numeral: an integer
//! keeps every digit, and a fraction has at most 308 integer digits. A
//! number with an exponent is admitted by its value. Its significand `S`
//! (the digits from the first nonzero one) and its order `m` (the power of
//! ten of that digit: the integer digits less one plus the exponent, or the
//! exponent less one less the zeros that lead a fraction below 1) decide it.
//! A value is finite when `m` is at most 307, so magnitudes from 1e308, just
//! below the largest finite 1.7976931348623157e308, are refused: conservative
//! by less than one decade. A minimum of `D` digits admits every order from
//! `D` up, and at order `D - 1` only a significand at least the minimum's
//! digits; a maximum of `D` digits admits every order below `D - 1`, and at
//! order `D - 1` only a significand at most its digits. Zero, signed or not,
//! is admitted under any exponent when the bounds contain it.
//!
//! The integer digits are counted at a nesting depth that does not grow with
//! their number, so the grammar stays within the wire's nesting bound.
//!
//! A finite automaton cannot weigh an unbounded digit count against an
//! unbounded exponent, and counting every digit for every bound would
//! multiply the automaton, so with an exponent these finite spellings are
//! refused:
//! - an integer part of more than 308 digits under a negative exponent;
//! - an integer part of more than [`WINDOW`] digits, for a sign whose values
//!   have a bound other than 0;
//! - a fraction below 1 with more than [`WINDOW`] leading zeros, unless the
//!   sign's values have no minimum above 0 and the exponent is at most the
//!   largest order plus [`WINDOW`] plus 2 (349 without a maximum);
//! - under an exclusive minimum other than 0, a value whose first digits are
//!   the minimum's followed by more integer digits.
//!
//! Each finite number is one rule that the automaton builds once and calls
//! from every place a number stands (`SchemaCompiler::finite_numbers`).

use std::collections::HashMap;

use crate::expr::{CharClass, Expr, GrammarBuilder, RuleId};
use crate::range::{Bound, number_range, same_length, unsigned_range};

/// The most integer digits of a number with a fraction or exponent: one
/// more is a magnitude of at least 1e308.
pub(crate) const INTEGER_DIGITS: u32 = 308;

/// The largest finite order this rule admits (below 1e308).
const LARGEST_ORDER: i64 = INTEGER_DIGITS as i64 - 1;

/// Leading fraction zeros, and integer digits under a bound, that are
/// counted exactly: well past a bound's 16 digits.
pub(crate) const WINDOW: u32 = 40;

fn class(low: u8, high: u8) -> Expr {
    Expr::Class(CharClass::from_ranges([(u32::from(low), u32::from(high))]))
}

fn digit() -> Expr {
    class(b'0', b'9')
}

fn nonzero() -> Expr {
    class(b'1', b'9')
}

fn literal(bytes: &[u8]) -> Expr {
    Expr::Literal(bytes.to_vec())
}

fn fraction() -> Expr {
    Expr::seq([Expr::literal("."), digit().plus()])
}

fn zero_fraction() -> Expr {
    Expr::seq([Expr::literal("."), Expr::literal("0").plus()])
}

/// Alternatives without [`Expr::alt`]'s deduplication, which hashes every
/// branch again at each level of a nested structure.
fn either(items: impl IntoIterator<Item = Expr>) -> Expr {
    let mut flat = Vec::new();
    for item in items {
        match item {
            Expr::Nothing => {}
            Expr::Alt(inner) => flat.extend(inner),
            item => flat.push(item),
        }
    }
    match flat.len() {
        0 => Expr::Nothing,
        1 => flat.pop().expect("one item"),
        _ => Expr::Alt(flat),
    }
}

/// `unit` read `count` times then `tail(count)`, for each `count` up to
/// `max`. Units nest within tens and tens within hundreds, so the nesting
/// depth does not grow with `max` and the size grows linearly.
fn counted(unit: &Expr, max: u32, tail: &mut dyn FnMut(u32) -> Expr) -> Expr {
    let mut hundreds = Expr::Nothing;
    for hundred in (0..=max / 100).rev() {
        let mut tens = Expr::Nothing;
        for ten in (0..10).rev() {
            let base = hundred * 100 + ten * 10;
            if base > max {
                continue;
            }
            let mut units = Expr::Nothing;
            for one in (0..10).rev() {
                if base + one <= max {
                    units = either([tail(base + one), Expr::seq([unit.clone(), units])]);
                }
            }
            tens = either([units, Expr::seq([unit.clone().repeat(10, Some(10)), tens])]);
        }
        hundreds = either([
            tens,
            Expr::seq([unit.clone().repeat(100, Some(100)), hundreds]),
        ]);
    }
    hundreds
}

/// Exponent pieces every integer length shares, as rules the automaton
/// builds once.
struct Numerals<'a> {
    builder: &'a mut GrammarBuilder,
    /// Any digits.
    digits: RuleId,
    /// A numeral below 100 without leading zeros.
    below_hundred: RuleId,
    /// Each exponent of one value.
    exact: HashMap<i64, RuleId>,
}

impl<'a> Numerals<'a> {
    fn new(builder: &'a mut GrammarBuilder) -> Self {
        let digits = builder.rule(digit().plus());
        let below_hundred = builder.rule(Expr::seq([nonzero().optional(), digit()]));
        Self {
            builder,
            digits,
            below_hundred,
            exact: HashMap::new(),
        }
    }

    /// Digits, possibly led by zeros, whose value lies in `low..=high`.
    fn padded(&self, low: i64, high: Option<i64>) -> Expr {
        match high {
            None if low == 0 => Expr::Rule(self.digits),
            Some(high) if low == 0 && high >= 100 => Expr::seq([
                Expr::literal("0").star(),
                either([
                    Expr::Rule(self.below_hundred),
                    same_length(b"100", high.to_string().as_bytes()),
                ]),
            ]),
            _ => Expr::seq([
                Expr::literal("0").star(),
                unsigned_range(low as u128, high.map(|high| high as u128)),
            ]),
        }
    }

    /// An exponent whose value lies in `low..=high`, each end `None` when
    /// unbounded.
    fn exponent(&self, low: Option<i64>, high: Option<i64>) -> Expr {
        if low.zip(high).is_some_and(|(low, high)| low > high) {
            return Expr::Nothing;
        }
        let mut signs = Vec::new();
        if high.is_none_or(|high| high >= 0) {
            let from = low.map_or(0, |low| low.max(0));
            signs.push(Expr::seq([
                Expr::literal("+").optional(),
                self.padded(from, high),
            ]));
        }
        if low.is_none_or(|low| low <= 0) {
            let from = high.map_or(0, |high| (-high).max(0));
            signs.push(Expr::seq([
                Expr::literal("-"),
                self.padded(from, low.map(|low| -low)),
            ]));
        }
        Expr::seq([Expr::Class(CharClass::of("eE")), either(signs)])
    }

    /// The exponent of `value`, one rule wherever a window ends in it.
    fn exactly(&mut self, value: i64) -> Expr {
        let rule = match self.exact.get(&value) {
            Some(&rule) => rule,
            None => {
                let exponent = self.exponent(Some(value), Some(value));
                let rule = self.builder.rule(exponent);
                self.exact.insert(value, rule);
                rule
            }
        };
        Expr::Rule(rule)
    }
}

/// A bound on a magnitude: its decimal digits and whether it is included.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Limit {
    digits: Vec<u8>,
    inclusive: bool,
}

impl Limit {
    fn new(value: u128, inclusive: bool) -> Self {
        Self {
            digits: value.to_string().into_bytes(),
            inclusive,
        }
    }

    fn len(&self) -> usize {
        self.digits.len()
    }
}

/// `digits` plus one, unless that takes another digit.
fn increment(digits: &[u8]) -> Option<Vec<u8>> {
    let mut next = digits.to_vec();
    for position in (0..next.len()).rev() {
        if next[position] < b'9' {
            next[position] += 1;
            return Some(next);
        }
        next[position] = b'0';
    }
    None
}

/// `digits` less one, unless that drops a digit.
fn decrement(digits: &[u8]) -> Option<Vec<u8>> {
    let mut next = digits.to_vec();
    for position in (0..next.len()).rev() {
        if next[position] > b'0' {
            next[position] -= 1;
            return (next[0] != b'0').then_some(next);
        }
        next[position] = b'9';
    }
    None
}

/// `digits` with a point after the first `point` of them.
fn pointed(digits: &[u8], point: Option<usize>) -> Vec<u8> {
    match point {
        Some(point) if point < digits.len() => {
            let mut text = digits[..point].to_vec();
            text.push(b'.');
            text.extend_from_slice(&digits[point..]);
            text
        }
        _ => digits.to_vec(),
    }
}

/// The first `low.len()` significant digits at a bound's order (the
/// window), a point after the first `point` of them or, when `None`, before
/// them all, then `exponent`. The window lies in `low..=high`, read as
/// followed by zeros when the significand is shorter. Any digits follow a
/// full window, except that one equal to `high` is followed only by zeros
/// when `tied`. Once the digits leave both `low` and `high` behind, any
/// digits follow.
struct Window<'a> {
    low: &'a [u8],
    high: &'a [u8],
    point: Option<usize>,
    tied: bool,
    exponent: Expr,
}

impl Window<'_> {
    /// The fewest digits a significand has: those before the point.
    fn shortest(&self) -> usize {
        self.point.unwrap_or(1)
    }

    /// Any digits after the first `at`, at least those before the point,
    /// then the exponent.
    fn rest(&self, at: usize) -> Expr {
        let exponent = self.exponent.clone();
        match self.point {
            Some(point) if at <= point => Expr::seq([
                digit().repeat((point - at) as u32, Some((point - at) as u32)),
                either([
                    exponent.clone(),
                    Expr::seq([Expr::literal("."), digit().plus(), exponent]),
                ]),
            ]),
            _ => Expr::seq([
                digit().repeat(self.shortest().saturating_sub(at) as u32, None),
                exponent,
            ]),
        }
    }

    fn read(&self, at: usize, on_low: bool, on_high: bool) -> Expr {
        if !on_low && !on_high {
            return self.rest(at);
        }
        if at == self.low.len() {
            let tail = if on_high && self.tied {
                Expr::literal("0").star()
            } else {
                digit().star()
            };
            return Expr::seq([tail, self.exponent.clone()]);
        }
        let mut branches = Vec::new();
        if at >= self.shortest() && (!on_low || self.low[at..].iter().all(|&digit| digit == b'0')) {
            branches.push(self.exponent.clone());
        }
        let first = if on_low { self.low[at] } else { b'0' };
        let last = if on_high { self.high[at] } else { b'9' };
        let mut digits = Vec::new();
        if on_low && on_high && first == last {
            digits.push(Expr::seq([
                literal(&[first]),
                self.read(at + 1, true, true),
            ]));
        } else if first <= last {
            let (mut from, mut to) = (first, last);
            if on_low {
                digits.push(Expr::seq([
                    literal(&[first]),
                    self.read(at + 1, true, false),
                ]));
                from += 1;
            }
            if on_high {
                digits.push(Expr::seq([
                    literal(&[last]),
                    self.read(at + 1, false, true),
                ]));
                to -= 1;
            }
            if from <= to {
                digits.push(Expr::seq([class(from, to), self.rest(at + 1)]));
            }
        }
        let digits = either(digits);
        branches.push(if self.point == Some(at) {
            Expr::seq([Expr::literal("."), digits])
        } else {
            digits
        });
        either(branches)
    }
}

/// Significands at order `len - 1` within the limits of that many digits,
/// each with the exponent that order takes.
fn boundary(
    numerals: &mut Numerals,
    len: usize,
    lower: Option<&Limit>,
    upper: Option<&Limit>,
) -> Expr {
    let order = len as i64 - 1;
    // Window values (the first `len` digits of the significand): the
    // least, the greatest with any digits after it, and the greatest read
    // as followed by zeros.
    let low = match lower {
        Some(lower) if lower.inclusive => Some(lower.digits.clone()),
        Some(lower) => increment(&lower.digits),
        None => Some([b"1".as_slice(), &vec![b'0'; len - 1]].concat()),
    };
    let (high_free, high_exact) = match upper {
        Some(upper) if upper.inclusive => (decrement(&upper.digits), Some(upper.digits.clone())),
        Some(upper) => (decrement(&upper.digits), decrement(&upper.digits)),
        None => (Some(vec![b'9'; len]), Some(vec![b'9'; len])),
    };
    let within = |high: &Option<Vec<u8>>| match (&low, high) {
        (Some(low), Some(high)) if low <= high => Some((low.clone(), high.clone())),
        _ => None,
    };
    let free = within(&high_free);
    let exact = within(&high_exact);
    // A window equal to an inclusive maximum, then only zeros.
    let tie_high = upper
        .filter(|upper| upper.inclusive)
        .filter(|upper| low.as_ref().is_some_and(|low| *low <= upper.digits))
        .map(|upper| upper.digits.clone());
    // A window equal to an exclusive minimum, then a nonzero digit after the
    // point.
    let tie_low = lower
        .filter(|lower| !lower.inclusive)
        .filter(|lower| upper.is_none_or(|upper| lower.digits < upper.digits))
        .map(|lower| lower.digits.clone());
    let more_than_zeros = || Expr::seq([Expr::literal("0").star(), nonzero(), digit().star()]);
    let mut parts = Vec::new();
    // The integer part holds the window, then more integer digits.
    let extra = WINDOW - len as u32;
    let mut shifted = |more: u32| numerals.exactly(order + 1 - len as i64 - i64::from(more));
    if let Some((low, high)) = &free {
        parts.push(Expr::seq([
            same_length(low, high),
            counted(&digit(), extra, &mut |more| {
                Expr::seq([fraction().optional(), shifted(more)])
            }),
        ]));
    }
    if let Some(tie) = &tie_high {
        parts.push(Expr::seq([
            literal(tie),
            counted(&Expr::literal("0"), extra, &mut |more| {
                Expr::seq([zero_fraction().optional(), shifted(more)])
            }),
        ]));
    }
    if let Some(tie) = &tie_low {
        parts.push(Expr::seq([
            literal(tie),
            Expr::literal("."),
            more_than_zeros(),
            shifted(0),
        ]));
    }
    // The point falls inside the window, after `point` digits, or before
    // it, after zeros; the window may end early.
    let window = |point: Option<usize>, exponent: Expr| {
        let mut parts = Vec::new();
        if let Some((low, high)) = &exact {
            parts.push(
                Window {
                    low,
                    high,
                    point,
                    tied: tie_high.is_some(),
                    exponent: exponent.clone(),
                }
                .read(0, true, true),
            );
        }
        if let Some(tie) = &tie_low {
            parts.push(Expr::seq([
                literal(&pointed(tie, point)),
                more_than_zeros(),
                exponent,
            ]));
        }
        either(parts)
    };
    for integers in 1..len {
        parts.push(window(
            Some(integers),
            numerals.exactly(order + 1 - integers as i64),
        ));
    }
    parts.push(Expr::seq([
        Expr::literal("0."),
        counted(&Expr::literal("0"), WINDOW, &mut |zeros| {
            window(None, numerals.exactly(order + 1 + i64::from(zeros)))
        }),
    ]));
    either(parts)
}

/// Nonzero magnitudes within the limits written with an exponent; a missing
/// upper limit admits every finite order.
fn magnitudes(numerals: &mut Numerals, lower: Option<&Limit>, upper: Option<&Limit>) -> Expr {
    // Orders every significand takes: from the minimum's length, to two
    // below the maximum's.
    let least = lower.map(|lower| lower.len() as i64);
    let greatest = upper.map_or(LARGEST_ORDER, |upper| upper.len() as i64 - 2);
    let mut parts = Vec::new();
    // Integer digits are counted to 308 without a bound, to [`WINDOW`]
    // with one.
    let counted_digits = if lower.is_none() && upper.is_none() {
        INTEGER_DIGITS
    } else {
        WINDOW
    };
    if least.is_none_or(|least| least <= greatest) {
        // `more + 1` integer digits take orders from `more`.
        parts.push(Expr::seq([
            nonzero(),
            counted(&digit(), counted_digits - 1, &mut |more| {
                let shift = i64::from(more);
                Expr::seq([
                    fraction().optional(),
                    numerals.exponent(least.map(|least| least - shift), Some(greatest - shift)),
                ])
            }),
        ]));
        // A fraction below 1 led by `zeros` zeros takes orders from
        // `-zeros - 1`.
        let mut fractions = vec![counted(&Expr::literal("0"), WINDOW, &mut |zeros| {
            let shift = i64::from(zeros) + 1;
            Expr::seq([
                nonzero(),
                digit().star(),
                numerals.exponent(least.map(|least| least + shift), Some(greatest + shift)),
            ])
        })];
        if least.is_none() {
            fractions.push(Expr::seq([
                Expr::literal("0").repeat(WINDOW + 1, None),
                nonzero(),
                digit().star(),
                numerals.exponent(None, Some(greatest + i64::from(WINDOW) + 2)),
            ]));
        }
        parts.push(Expr::seq([Expr::literal("0."), either(fractions)]));
    }
    match (lower, upper) {
        (Some(lower), Some(upper)) if lower.len() == upper.len() => {
            parts.push(boundary(numerals, lower.len(), Some(lower), Some(upper)));
        }
        _ => {
            if let Some(lower) = lower {
                parts.push(boundary(numerals, lower.len(), Some(lower), None));
            }
            if let Some(upper) = upper {
                parts.push(boundary(numerals, upper.len(), None, Some(upper)));
            }
        }
    }
    either(parts)
}

/// Whether the magnitude limits leave no value.
fn empty(lower: Option<&Limit>, upper: Option<&Limit>) -> bool {
    let (Some(lower), Some(upper)) = (lower, upper) else {
        return false;
    };
    let order = |limit: &Limit| (limit.len(), limit.digits.clone());
    order(lower) > order(upper)
        || (order(lower) == order(upper) && !(lower.inclusive && upper.inclusive))
}

/// JSON numbers between the bounds whose value a binary64 holds as a finite
/// number, by the rules in the module documentation. Pieces the automaton
/// builds once are rules of `builder`.
pub(crate) fn finite_number(
    builder: &mut GrammarBuilder,
    lower: Option<Bound>,
    upper: Option<Bound>,
) -> Expr {
    let mut numerals = Numerals::new(builder);
    let contains_zero = lower
        .is_none_or(|lower| lower.value < 0 || (lower.value == 0 && lower.inclusive))
        && upper.is_none_or(|upper| upper.value > 0 || (upper.value == 0 && upper.inclusive));
    // Positive values: above a positive minimum, below a maximum.
    let positive = match upper {
        Some(upper) if upper.value <= 0 => None,
        _ => Some((
            lower
                .filter(|lower| lower.value > 0)
                .map(|lower| Limit::new(lower.value as u128, lower.inclusive)),
            upper.map(|upper| Limit::new(upper.value as u128, upper.inclusive)),
        )),
    };
    // Negative values by magnitude: above a negative maximum's, below a
    // minimum's.
    let negative = match lower {
        Some(lower) if lower.value >= 0 => None,
        _ => Some((
            upper
                .filter(|upper| upper.value < 0)
                .map(|upper| Limit::new(upper.value.unsigned_abs(), upper.inclusive)),
            lower.map(|lower| Limit::new(lower.value.unsigned_abs(), lower.inclusive)),
        )),
    };
    let mut side = |limits: &Option<(Option<Limit>, Option<Limit>)>| match limits {
        Some((lower, upper)) if !empty(lower.as_ref(), upper.as_ref()) => {
            magnitudes(&mut numerals, lower.as_ref(), upper.as_ref())
        }
        _ => Expr::Nothing,
    };
    let mut parts = vec![number_range(lower, upper, Some(INTEGER_DIGITS))];
    if positive == negative {
        parts.push(Expr::seq([Expr::literal("-").optional(), side(&positive)]));
    } else {
        parts.push(side(&positive));
        parts.push(Expr::seq([Expr::literal("-"), side(&negative)]));
    }
    if contains_zero {
        parts.push(Expr::seq([
            Expr::literal("-").optional(),
            Expr::literal("0"),
            zero_fraction().optional(),
            Expr::seq([
                Expr::Class(CharClass::of("eE")),
                Expr::Class(CharClass::of("+-")).optional(),
                digit().plus(),
            ])
            .optional(),
        ]));
    }
    either(parts)
}
