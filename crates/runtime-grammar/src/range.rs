//! Decimal numerals bounded by integers, as exact expressions.

use crate::expr::{CharClass, Expr};

fn digit_class(lo: u8, hi: u8) -> Expr {
    Expr::Class(CharClass::from_ranges([(u32::from(lo), u32::from(hi))]))
}

fn any_digits(count: usize) -> Expr {
    digit_class(b'0', b'9').repeat(count as u32, Some(count as u32))
}

/// Digit strings of `low.len()` digits between `low` and `high` inclusive.
fn same_length(low: &[u8], high: &[u8]) -> Expr {
    let Some((&first_low, rest_low)) = low.split_first() else {
        return Expr::Empty;
    };
    let (&first_high, rest_high) = high.split_first().expect("same length");
    if first_low == first_high {
        return Expr::seq([
            Expr::Literal(vec![first_low]),
            same_length(rest_low, rest_high),
        ]);
    }
    let mut branches = vec![Expr::seq([
        Expr::Literal(vec![first_low]),
        at_least(rest_low),
    ])];
    if first_high - first_low > 1 {
        branches.push(Expr::seq([
            digit_class(first_low + 1, first_high - 1),
            any_digits(rest_low.len()),
        ]));
    }
    branches.push(Expr::seq([
        Expr::Literal(vec![first_high]),
        at_most(rest_high),
    ]));
    Expr::alt(branches)
}

fn at_least(low: &[u8]) -> Expr {
    let Some((&first, rest)) = low.split_first() else {
        return Expr::Empty;
    };
    if low.iter().all(|&digit| digit == b'0') {
        return any_digits(low.len());
    }
    let mut branches = vec![Expr::seq([Expr::Literal(vec![first]), at_least(rest)])];
    if first < b'9' {
        branches.push(Expr::seq([
            digit_class(first + 1, b'9'),
            any_digits(rest.len()),
        ]));
    }
    Expr::alt(branches)
}

fn at_most(high: &[u8]) -> Expr {
    let Some((&first, rest)) = high.split_first() else {
        return Expr::Empty;
    };
    if high.iter().all(|&digit| digit == b'9') {
        return any_digits(high.len());
    }
    let mut branches = vec![Expr::seq([Expr::Literal(vec![first]), at_most(rest)])];
    if first > b'0' {
        branches.push(Expr::seq([
            digit_class(b'0', first - 1),
            any_digits(rest.len()),
        ]));
    }
    Expr::alt(branches)
}

/// Canonical numerals (no leading zeros) for integers in `low..=high`.
pub(crate) fn unsigned_range(low: u128, high: Option<u128>) -> Expr {
    unsigned_range_digits(low, high, None)
}

/// The numerals of [`unsigned_range`] of at most `max_digits` digits.
fn unsigned_range_digits(low: u128, high: Option<u128>, max_digits: Option<u32>) -> Expr {
    if high.is_some_and(|high| high < low) {
        return Expr::Nothing;
    }
    let low_text = low.to_string().into_bytes();
    let mut branches = Vec::new();
    let last_length = high.map_or(low_text.len(), |high| high.to_string().len());
    for length in low_text.len()..=last_length {
        let start = if length == low_text.len() {
            low
        } else {
            10_u128.pow(length as u32 - 1)
        };
        let end = 10_u128.pow(length as u32) - 1;
        let end = high.map_or(end, |high| high.min(end));
        branches.push(same_length(
            start.to_string().as_bytes(),
            end.to_string().as_bytes(),
        ));
    }
    let longer = max_digits.is_none_or(|digits| digits as usize > low_text.len());
    if high.is_none() && longer {
        branches.push(Expr::seq([
            digit_class(b'1', b'9'),
            digit_class(b'0', b'9')
                .repeat(low_text.len() as u32, max_digits.map(|digits| digits - 1)),
        ]));
    }
    Expr::alt(branches)
}

/// One end of a range: the bound and whether it is included.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Bound {
    pub(crate) value: i128,
    pub(crate) inclusive: bool,
}

/// Integers between the bounds.
pub(crate) fn integer_range(lower: Option<Bound>, upper: Option<Bound>) -> Expr {
    let low = lower.map(|bound| bound.value + i128::from(!bound.inclusive));
    let high = upper.map(|bound| bound.value - i128::from(!bound.inclusive));
    let mut branches = Vec::new();
    if high.is_none_or(|high| high >= 0) {
        let from = low.map_or(0, |low| low.max(0)) as u128;
        branches.push(unsigned_range(from, high.map(|high| high as u128)));
    }
    if low.is_none_or(|low| low < 0) {
        let from = high.map_or(1, |high| (-high).max(1)) as u128;
        let to = low.map(|low| (-low) as u128);
        branches.push(Expr::seq([Expr::literal("-"), unsigned_range(from, to)]));
    }
    Expr::alt(branches)
}

fn fraction_any() -> Expr {
    Expr::seq([Expr::literal("."), digit_class(b'0', b'9').plus()]).optional()
}

fn fraction_zero() -> Expr {
    Expr::seq([Expr::literal("."), Expr::literal("0").plus()]).optional()
}

fn fraction_nonzero() -> Expr {
    Expr::seq([
        Expr::literal("."),
        digit_class(b'0', b'9').star(),
        digit_class(b'1', b'9'),
        digit_class(b'0', b'9').star(),
    ])
}

/// Non-negative decimals without exponent between integer bounds on their
/// magnitude, an unbounded one of at most `max_digits` integer digits.
fn magnitude_range(lower: Bound, upper: Option<Bound>, max_digits: Option<u32>) -> Expr {
    let low = lower.value as u128;
    let mut branches = Vec::new();
    match upper {
        Some(upper) if upper.value < lower.value => return Expr::Nothing,
        Some(upper) if upper.value == lower.value => {
            if lower.inclusive && upper.inclusive {
                branches.push(Expr::seq([unsigned_range(low, Some(low)), fraction_zero()]));
            }
            return Expr::alt(branches);
        }
        _ => {}
    }
    let equal_low = if lower.inclusive {
        fraction_any()
    } else {
        fraction_nonzero()
    };
    branches.push(Expr::seq([unsigned_range(low, Some(low)), equal_low]));
    match upper {
        Some(upper) => {
            let high = upper.value as u128;
            if high > low + 1 {
                branches.push(Expr::seq([
                    unsigned_range(low + 1, Some(high - 1)),
                    fraction_any(),
                ]));
            }
            if upper.inclusive {
                branches.push(Expr::seq([
                    unsigned_range(high, Some(high)),
                    fraction_zero(),
                ]));
            }
        }
        None => branches.push(Expr::seq([
            unsigned_range_digits(low + 1, None, max_digits),
            fraction_any(),
        ])),
    }
    Expr::alt(branches)
}

/// JSON numbers without exponent between integer bounds; one that a bound
/// leaves unbounded has at most `max_digits` integer digits.
pub(crate) fn number_range(
    lower: Option<Bound>,
    upper: Option<Bound>,
    max_digits: Option<u32>,
) -> Expr {
    let mut branches = Vec::new();
    let nonnegative =
        upper.is_none_or(|upper| upper.value > 0 || (upper.value == 0 && upper.inclusive));
    if nonnegative {
        let low = match lower {
            Some(lower) if lower.value > 0 || (lower.value == 0 && !lower.inclusive) => lower,
            _ => Bound {
                value: 0,
                inclusive: true,
            },
        };
        branches.push(magnitude_range(low, upper, max_digits));
    }
    let negative = lower.is_none_or(|lower| lower.value < 0);
    if negative {
        let low = match upper {
            Some(upper) if upper.value < 0 => Bound {
                value: -upper.value,
                inclusive: upper.inclusive,
            },
            _ => Bound {
                value: 0,
                inclusive: false,
            },
        };
        let high = lower.map(|lower| Bound {
            value: -lower.value,
            inclusive: lower.inclusive,
        });
        branches.push(Expr::seq([
            Expr::literal("-"),
            magnitude_range(low, high, max_digits),
        ]));
    }
    Expr::alt(branches)
}
